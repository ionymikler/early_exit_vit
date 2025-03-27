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
        def true_fn(x):
            x = x + 1.0
            return x

        def false_fn(x):
            x = x - 1.0
            return x

        x = torch.cond(
            pred=x.sum() > 0, true_fn=true_fn, false_fn=false_fn, operands=(x,)
        )
        return x


input_tensor_1 = torch.tensor([1, 2])
input_tensor_2 = torch.tensor([1, -2])
model = Wrapper()


result = model(input_tensor_1)
print(result)
# result = model(input_tensor_2)
# print(result)

# print("Exporting program...")
# ep = torch.export.export(model, (input_tensor_1,), strict=False)
# # print(ep)
print("Program exported")
print("Exporting to ONNX...")
onnx_program = torch.onnx.export(
    model,
    # ep,
    (input_tensor_1,),
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
    fallback=False,
    report=True,
)
onnx_program.save("example_conditional_2.onnx")
