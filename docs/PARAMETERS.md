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

- **固定高度 48** 直接来自识别模型输入的第 3 维，不需要查文档；库把它作为 `Config.Height` 的默认值，模型换成静态高度不同的变体会在启动时被校验。
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
| 通道顺序 | RGB | RGB | PaddleOCR v5/v6 推理配置（`order: RGB`） |
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

探测每次输入尺寸只发生一次，随后命中缓存；一次探测失败的推理开销是毫秒级。

## 5. 后处理参数（DB 算法常量）

检测模型输出的是"文字概率图"，DB（Differentiable Binarization）后处理把概率图变成文本框：

| 参数 | 默认值 | 来源/理由 |
|---|---|---|
| 二值化阈值 | 0.3 | PaddleOCR DB 后处理默认 `thresh=0.3` |
| 最小框面积 | 10 px | 过滤噪点 |
| unclip 比例 | 1.5 | DB 训练时按 shrink ratio 收缩了文本区域，推理端按 `d = area*ratio/perimeter` 向外膨胀补偿 |

连通域用 8 邻域并查集标记，同行相邻框按"垂直重叠超过较短框一半、水平间隙不超过框高"合并成整行，再交给识别模型。

## 6. 置信度计算的一个坑

PP-OCRv6 的 ONNX 导出**末端包含 Softmax 节点**（计算图里能看到 `Softmax.2`），所以模型输出已经是概率分布。如果按常规做法再 softmax 一次，分布会被压平成近似均匀分布：argmax 不变（文字仍然对），但置信度从 0.98 掉到 0.0001。库里的 `looksLikeProbabilities` 检测行和是否接近 1，只对真正的 logits 输出做 softmax。

## 7. 参数封装方式

- 所有参数集中在 `Config` / `DetConfig` 两个结构体，零值自动填默认值（`applyDefaults`），路径等必需项单独校验（`normalize`），两部分分开是为了让检测阶段复用归一化默认值而不触发路径校验。
- 模型能自省的参数（张量名、高度、类别数）不暴露给用户，加载时自动读取；框架决定的常量（均值、阈值、unclip）暴露为可覆盖字段，默认值写在字段注释里。
- 全部参数都有测试覆盖：归一化数值有精确断言（`preprocess_test.go`），解码映射有构造用例（`decode_test.go`），端到端有 golden 测试（`rec_test.go`、`pipeline_test.go`）。
