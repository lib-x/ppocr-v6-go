# 参数获取与封装记录

本文记录 `ppocr-v6-go` 用到的每个模型参数从哪里来、怎么验证、在库里如何封装。写作目标是让没有 OCR 背景的人也能复现同样的过程。

## 1. 模型文件本身能读出来的参数（自动自省）

ONNX 文件里存着计算图，图的输入/输出声明就是一份现成的接口文档。库在 `New()` / `NewDetector()` 时直接解析这些声明（`internal/onnxmeta`，手写的 protobuf wire-format 读取器，不引入 protobuf 运行时）：

| 参数 | 来源 | 识别模型实测值 | 检测模型实测值 |
|---|---|---|---|
| 输入张量名 | `graph.input[].name` | `x` | `x` |
| 输入形状 | `graph.input[].type.tensor_type.shape` | `[batch, 3, 48, width]` | `[batch, 3, height, width]` |
| 输出张量名 | `graph.output[].name` | `fetch_name_0` | `fetch_name_0` |
| 输出形状 | `graph.output[].type.tensor_type.shape` | `[batch, T, 18710]` | `[batch, 1, height, width]` |
| 动态维度 | `dim.dim_param`（如 `DynamicDimension.1`） | batch 与 width 动态 | 全部动态 |

要点：

- **固定高度 48** 是识别模型的静态输入高度，库用它当 `Config.Height` 的默认值（写在字段注释里，可以覆盖），并在启动时拿图里的第 3 维做校验，高度不匹配会直接报错。检测侧的输入高度是动态的，没有这个约束。
- **输出类别数 18710** 来自识别模型输出最后一维，库用它校验字典大小（见第 3 节）。
- **动态维度**表示这些轴在运行时确定。ONNX Runtime 要求调用方提供**精确匹配**的输出张量形状，所以库用"形状探测"获取真实尺寸（见第 4 节）。

复现命令（Python + onnx 包）：

```python
import onnx
m = onnx.load("inference.onnx", load_external_data=False)
g = m.graph
for v in list(g.input) + list(g.output):
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in v.type.tensor_type.shape.dim]
    print(v.name, dims)
```

## 2. 预处理参数（模型文件里没有，来自 PaddleOCR 源码）

ONNX 只描述网络计算，不描述图像怎么变成输入张量。这部分参数必须从训练/推理框架的代码里取：

| 参数 | 识别（rec） | 检测（det） | 来源 |
|---|---|---|---|
| 归一化 | `(pixel/255 - 0.5) / 0.5` | `(pixel/255 - mean) / std`，mean=`[0.485,0.456,0.406]`，std=`[0.229,0.224,0.225]`（ImageNet） | PaddleOCR `rec_img_aug.py` / det 预处理配置 |
| 通道顺序 | RGB（默认，`Config.BGR` 可切换） | RGB（固定） | PaddleX 推理链路 `ReadImage(format="RGB")` |
| 缩放策略 | 高度固定 48，宽度按比例 `ceil(48*w/h)` | 长边不超过 960，等比缩放后补白到 32 的倍数 | `DetResizeForTest`（`limit_side_len=960`） |
| 插值 | 双线性 | 双线性 | `cv2.INTER_LINEAR` |

验证方式：识别链路的 golden 测试（`testdata/test3.png` 是一张 669x45 的截图）识别结果与图片文字完全一致，说明归一化/通道顺序/缩放三者都正确。检测侧还有一个辅助证据：把检测的归一化误用成识别参数时，概率图明显变弱、框体断裂；换成 ImageNet 参数后响应稳定。

## 3. 字典与类别映射（模型仓库之外）

识别模型输出 18710 个类别，但字典文件不在模型仓库里。获取路径：

1. 字典文件：`PaddlePaddle/PaddleOCR` 仓库 `ppocr/utils/dict/ppocrv6_dict.txt`，18708 行，每行一个字符（含 emoji）。
2. 类别布局：18710 = **1（CTC blank）+ 18708（字典）+ 1（空格）**。空格的存在由训练配置 `use_space_char: true` 确认（`configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml`）。
3. 解码映射：类别 0 是 blank（解码时丢弃并重置重复计数）；类别 `1..18708` 映射到字典第 `class-1` 个字符；类别 18709 是空格。

库的处理：字典通过 `go:embed` 内嵌（`DefaultDictBytes`），`New()` 用模型的类别数校验字典大小，不一致直接报错（防止拿错字典静默产生乱码）。

## 4. 运行时才能确定的形状（形状探测）

两个模型的输出都有动态维度，而 ONNX Runtime 的会话输出要求形状精确匹配：

- **识别模型**：输入宽度 W 决定 CTC 序列长度 T（`T = f(W)`，与骨干网络的步长有关，本模型约 `W/8`）。库先用一个故意超大的输出张量运行一次，让 ORT 报错并给出真实形状，解析后按宽度缓存（`Recognizer.outputLen`）。
- **检测模型**：输出是**全分辨率**概率图（末端是 ConvTranspose 上采样，输出等于输入尺寸）。这里有个陷阱：检测图**不会**校验超大输出张量（它会静默地把数据写进缓冲区前段），必须用故意过小的张量（`[1,1,1,1]`）逼它报错才能拿到真实形状（`Detector.probeOutputSize`）。

探测按输入尺寸缓存（识别按宽度，检测按 `(w, h)`），同一尺寸只发生一次；一次探测失败的推理开销是毫秒级。

## 5. 后处理参数（DB 算法常量）

检测模型输出的是"文字概率图"，DB（Differentiable Binarization）后处理把概率图变成文本框：

| 参数 | 默认值 | 参考实现的对应值 | 来源/理由 |
|---|---|---|---|
| 二值化阈值 `Threshold` | 0.3 | 0.3（PaddleX `thresh`） | 概率图是逐像素的"文字概率"，0.3 偏向召回：先多留候选，误检交给框级过滤兜底 |
| 框级阈值 `BoxThresh` | 0.6 | 0.6（PaddleX 部署默认；v6 训练配置写 0.45） | 计算框内概率均值，低于阈值整框丢弃，用来滤掉"零星亮、整体不像文字"的误检 |
| 最小框面积 `MinBoxArea` | 10 px | 参考用短边 `min_size=3` | 去掉二值化后的噪点小块，二者效果接近 |
| 最小边长 `MinBoxSide` | 3 px | 3（unclip 后另判 5） | 同上，按短边过滤细碎框 |
| unclip 比例 `UnclipRatio` | 1.5 | 训练配置 1.4 / 老推理链路 1.5 / PaddleX 2.0 | DB 训练时把文字区域按 shrink_ratio 0.4 缩小后再让模型学，推理端按 `d = area*ratio/perimeter` 放大补偿。官方链路里三个值并存，本库取 1.5，实测比 2.0 的识别均分高（0.9489 对 0.9347），且都可配置 |

连通域用 8 邻域并查集标记（返回的编号必须紧凑，否则会造出幽灵框，见代码里的回归测试），同行相邻框按"垂直重叠超过较短框一半、水平间隙不超过框高"合并成整行，再交给识别模型。参考实现不做这步合并（它用旋转框，靠排序组织行），本库的框是轴对齐的，同一行常被切成"菜名""虚线""价格"多段，所以补了这步；间隙阈值取框高，是因为同行内部的空隙（空格、点线）通常小于行高，而跨列的距离通常更大。

## 5b. 每个系数为什么是这个值

上表给了来源，这里补"为什么"和"改动的后果"，方便调参时判断方向。

| 参数 | 为什么是这个值 | 改大/改小的后果 |
|---|---|---|
| 检测 `MaxSideLen` 960 | 官方对 PP-OCRv6 的部署配置是长边 960（`resize_long=960`）。图越大，小字召回越好，但耗时按面积增长，960 是折中 | 调大：小字更稳、更慢；调小：快，密集小字漏检 |
| 补白到 32 的倍数 | 检测网络的总下采样步长是 32（每 32 像素汇总成一个特征点），尺寸不是 32 的倍数时最后一格不完整。参考实现是把尺寸 round 到 32 的倍数，本库改成向右下补白，避免为对齐而轻微拉伸图像 | 不补齐：动态形状下仍能跑，但边缘特征不完整，靠近右下的文字容易漏 |
| 检测归一化 ImageNet 均值方差 | 检测骨干按 ImageNet 统计量训练，输入分布必须一致 | 换成识别用的 0.5/0.5：概率图整体变弱、条带断裂 |
| 识别高度 48 | 训练配置 `d2s_train_image_shape: [3,48,320]`，模型按高 48 学到的字形模式 | 改高度：轻则精度下降，重则构造期校验直接报错 |
| 识别宽度 `ceil(48*w/h)` | 等比缩放保持字形比例，不拉伸 | 固定宽度拉伸：字形变形，长行准确率下降明显 |
| 识别归一化 `(x/255-0.5)/0.5` | 把 0..255 映射到 -1..1，训练时的数值范围 | 用检测的 ImageNet 参数：识别结果基本崩掉 |
| CTC blank = 0、空格 = 18709 | 字典 18708 行，加 blank 和空格共 18710 类，和模型输出最后一维一致 | 映射错位：整段文字变成乱码，所以构造时会校验类别数 |
| 置信度取均值 | 和 PaddleOCR `CTCLabelDecode` 一致（`np.mean(conf_list)`） | 换成概率乘积：长文本的分数会被指数级压低，阈值失去可比性 |
| 形状探测缓存 | ONNX Runtime 要求输出形状精确匹配，探测要跑一次"故意失败"的推理 | 不缓存：每次调用多一次毫秒级失败推理，批量处理时浪费明显 |

本库与参考实现的两处有意偏差，记录在这里：

- 参考实现的 `limit_type='max'` 只在长边超过 960 时缩小，小图保持原尺寸；本库统一把长边缩放到 960，小图会被放大。代价是小图多花算力，收益是小字获得更多像素。
- 参考实现把缩放后的尺寸 round 到 32 的倍数（会轻微拉伸）；本库保持等比、向右下补白（补白用白色，贴合常见白底文档）。

## 6. 置信度计算的一个坑

PP-OCRv6 的 ONNX 导出**末端包含 Softmax 节点**（计算图里能看到 `Softmax.2`），所以模型输出已经是概率分布。如果按常规做法再 softmax 一次，分布会被压平成近似均匀分布：argmax 不变（文字仍然对），但置信度从 0.98 掉到 0.0001。库里的 `looksLikeProbabilities` 检测行和是否接近 1，只对真正的 logits 输出做 softmax。

## 7. 参数封装方式

- 所有参数集中在 `Config` / `DetConfig` 两个结构体，零值自动填默认值（`applyDefaults`），路径等必需项单独校验（`normalize`），两部分分开是为了让检测阶段复用归一化默认值而不触发路径校验。
- 检测侧的张量名从图里自动读取；识别侧的张量名和高度有模型对应的默认值（`Config.InputName` / `OutputName` / `Height` 可覆盖），类别数从图里读出来用于校验字典。
- 框架决定的常量（识别均值/标准差、检测阈值、框级阈值、最小边长、unclip 比例、检测长边上限）暴露为可覆盖字段，默认值写在字段注释里；检测的归一化均值/标准差目前是固定值，没有导出成配置项。
- 全部参数都有测试覆盖：归一化数值有精确断言（`preprocess_test.go`），解码映射有构造用例（`decode_test.go`），端到端有 golden 测试（`rec_test.go`、`pipeline_test.go`）。

## 8. 示意图怎么重新生成

`docs/det-overlay.png`（检测框叠加）和 `docs/det-probmap.png`（原始概率图）由 `det_figures_test.go` 生成，默认跳过，需要时显式开启：

```bash
PPOCR_UPDATE_FIGURES=1 ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so go test -run TestGenerateDocsFigures .
```
