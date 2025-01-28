import torch


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_model = CondModel()

    def forward(self, x):
        nt = self.cond_model(x)
        return nt


class CondModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.ones_like(x)

        def true_fn(x, z):
            x = x + 1.0
            z = z * 1.0
            return x, z

        def false_fn(x, z):
            x = x - 1.0
            z = z * 0.0
            return x, z

        x = torch.cond(x.sum() > 0, true_fn, false_fn, (x, z))
        return x, z


input_tensor_1 = torch.tensor([1, 2])
input_tensor_2 = torch.tensor([1, -2])
model = Wrapper()


result = model(input_tensor_1)
print(result)
# result = model(input_tensor_2)
# print(result)

print("Exporting program...")
ep = torch.export.export(model, (input_tensor_1,), strict=False)
# print(ep)
print("Program exported")
print("Exporting to ONNX...")
onnx_program = torch.onnx.export(
    # model,
    ep,
    (input_tensor_1,),
    dynamo=True,
    fallback=False,
    report=True,
)
onnx_program.save("example_conditional.onnx")
