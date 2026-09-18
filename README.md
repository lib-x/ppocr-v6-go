# ppocr-v6-go

PP-OCRv6 文本检测 + 识别的纯 Go 封装，基于 [pure-onnx](https://github.com/amikos-tech/pure-onnx)（purego 调用 ONNX Runtime，无需 CGo）。

```go
import ppocrv6 "github.com/lib-x/ppocr-v6-go"

// 整图 OCR：检测 + 识别
p, err := ppocrv6.NewPipeline(
    ppocrv6.Config{ModelPath: "inference.onnx"},        // PP-OCRv6_small_rec_onnx
    ppocrv6.DetConfig{ModelPath: "det-v6.onnx"},        // PP-OCRv6_medium_det_onnx
)
if err != nil { log.Fatal(err) }
defer p.Close()

results, err := p.Recognize(img)
for _, r := range results {
    fmt.Printf("%s (%.3f)\n", r.Text, r.Score)
}

// 单行识别（已有裁剪好的文字行）
rec, err := ppocrv6.New(ppocrv6.Config{ModelPath: "inference.onnx"})
defer rec.Close()
res, err := rec.Recognize(lineImg)
```

## 特性

- **检测 + 识别全流程**：整图进，文字行（含坐标与置信度）出
- **字典内嵌**：`ppocrv6_dict.txt`（18708 字符）编译进二进制，只需模型文件
- **模型自省**：加载时解析 ONNX 图，校验输入形状与类别数；检测侧直接读取张量名，识别侧使用模型对应的默认张量名与高度（可覆盖）
- **动态形状探测**：CTC 序列长度、检测概率图尺寸按输入尺寸自动标定并缓存
- **纯 Go 依赖**：无 CGo；ONNX Runtime 共享库通过 `ONNXRUNTIME_LIB_PATH` 指定或由 pure-onnx 自动下载

## 模型文件

| 用途 | 模型 | 来源 |
|---|---|---|
| 识别 | PP-OCRv6_small_rec_onnx `inference.onnx` | [ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv6_small_rec_onnx) |
| 检测 | PP-OCRv6_medium_det_onnx `inference.onnx` | [HuggingFace](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx) |

## 运行环境

ONNX Runtime 共享库（1.24.x，与 pure-onnx 的 C API 版本对齐）：

```bash
# 方式一：指定已有共享库
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so.1.24.1

# 方式二：不设置，由 pure-onnx bootstrap 自动下载并缓存
```

## 配置

`Config`（识别）与 `DetConfig`（检测）的零值自动填充模型验证过的默认值；关键参数（均值/标准差、阈值、unclip 比例等）均可覆盖，详见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。

## CLI

```bash
go run ./cmd/ppocrv6 -model inference.onnx -dict "" -det det-v6.onnx page.png
```

不带 `-det` 时按单行识别处理；带 `-det` 时按整图 OCR 输出所有行。

## License

Apache-2.0（内嵌字典来自 PaddleOCR，Apache-2.0）。
