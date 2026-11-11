import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# =========================================================
# 1. 定义一个简单模型（替换成你的真实模型即可）
# =========================================================
class MLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_model():
    return MLP()


def make_synthetic_data(num_samples=4000, in_dim=768, num_classes=10):
    """生成以不同类别中心为均值的高斯簇，并划分训练集和测试集。"""
    generator = torch.Generator().manual_seed(42)
    centers = torch.randn(num_classes, in_dim, generator=generator) * 2.0
    labels = torch.randint(num_classes, (num_samples,), generator=generator)
    features = centers[labels] + 0.6 * torch.randn(
        num_samples, in_dim, generator=generator)

    order = torch.randperm(num_samples, generator=generator)
    features, labels = features[order], labels[order]
    split = int(num_samples * 0.8)
    return features[:split], labels[:split], features[split:], labels[split:]


def train_model(model, train_x, train_y, test_x, test_y,
                epochs=8, batch_size=256, learning_rate=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(train_x.size(0))
        total_loss = 0.0
        for start in range(0, train_x.size(0), batch_size):
            index = order[start:start + batch_size]
            loss = criterion(model(train_x[index]), train_y[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * index.size(0)

        model.eval()
        with torch.no_grad():
            predictions = model(test_x).argmax(dim=1)
            accuracy = (predictions == test_y).float().mean().item()
        print(f"[train] epoch {epoch + 1}/{epochs} "
              f"loss={total_loss / train_x.size(0):.4f} test_acc={accuracy:.2%}")

    model.eval()


# =========================================================
# 2. 导出 ONNX
# =========================================================
def export_onnx(model, path="model.onnx", dynamic=False):
    dummy = torch.randn(1, 768)

    dynamic_axes = None
    if dynamic:
        # 允许 batch 维动态，适合线上变长 batch
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,               # 避免新版 Gemm 图与量化器的形状推断不兼容
        do_constant_folding=True,   # 常量折叠，图优化的一部分
    )
    print(f"[export] saved to {path}, dynamic={dynamic}")


# =========================================================
# 3. PyTorch 推理计时
# =========================================================
@torch.no_grad()
def bench_torch(model, x, warmup=10, iters=100):
    x_t = torch.from_numpy(x)
    for _ in range(warmup):
        model(x_t)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x_t)
    dt = time.perf_counter() - t0
    return dt / iters * 1000  # ms/次


# =========================================================
# 4. onnxruntime 推理计时
# =========================================================
def bench_ort(sess, x, warmup=10, iters=100):
    input_name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {input_name: x})
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, {input_name: x})
    dt = time.perf_counter() - t0
    return dt / iters * 1000


# =========================================================
# 5. 构建优化后的 session
# =========================================================
def make_session(path, providers=None, opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                 intra_op_num_threads=4, inter_op_num_threads=2, parallel=False):
    so = ort.SessionOptions()
    so.graph_optimization_level = opt_level
    so.intra_op_num_threads = intra_op_num_threads  # 单个算子内部使用的线程数
    so.execution_mode = (ort.ExecutionMode.ORT_PARALLEL if parallel
                         else ort.ExecutionMode.ORT_SEQUENTIAL)
    if parallel:
        # 仅并行执行时生效：控制可独立执行的算子之间的并行度
        so.inter_op_num_threads = inter_op_num_threads

    if providers is None:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(path, sess_options=so, providers=providers)


# =========================================================
# 6. 主流程
# =========================================================
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model = build_model()
    train_x, train_y, test_x, test_y = make_synthetic_data()
    train_model(model, train_x, train_y, test_x, test_y)

    # ---------- 6.1 导出静态 shape ----------
    export_onnx(model, "./model_static.onnx", dynamic=False)
    # ---------- 6.2 导出动态 batch ----------
    export_onnx(model, "./model_dynamic.onnx", dynamic=True)

    # 取一条测试样本做推理与输出一致性验证
    x = test_x[:1].numpy().astype(np.float32)

    # ---------- 6.3 PyTorch 基线 ----------
    t_torch = bench_torch(model, x)
    print(f"[torch]        {t_torch:.3f} ms/iter")

    # ---------- 6.4 静态 ONNX + 默认优化 ----------
    sess_static = make_session("./model_static.onnx")
    t_static = bench_ort(sess_static, x)
    print(f"[onnx static]  {t_static:.3f} ms/iter")

    # ---------- 6.5 动态 ONNX ----------
    sess_dyn = make_session("./model_dynamic.onnx")
    t_dyn = bench_ort(sess_dyn, x)
    print(f"[onnx dynamic] {t_dyn:.3f} ms/iter")

    # ---------- 6.6 INT8 动态量化 ----------
    quantize_dynamic(
        "./model_static.onnx",
        "model_int8.onnx",
        weight_type=QuantType.QInt8,
    )
    sess_int8 = make_session("model_int8.onnx")
    t_int8 = bench_ort(sess_int8, x)
    print(f"[onnx int8]    {t_int8:.3f} ms/iter")

    # ---------- 6.7 验证输出一致性 ----------
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out_static = sess_static.run(None, {sess_static.get_inputs()[0].name: x})[0]
    out_int8 = sess_int8.run(None, {sess_int8.get_inputs()[0].name: x})[0]

    print("\n[check] static max abs diff:", np.abs(ref - out_static).max())
    print("[check] int8   max abs diff:", np.abs(ref - out_int8).max())

    # ---------- 6.8 汇总 ----------
    print("\n===== summary (ms/iter) =====")
    print(f"torch        : {t_torch:.3f}")
    print(f"onnx static  : {t_static:.3f}")
    print(f"onnx dynamic : {t_dyn:.3f}")
    print(f"onnx int8    : {t_int8:.3f}")


if __name__ == "__main__":
    main()