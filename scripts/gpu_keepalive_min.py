import torch, time
x = torch.zeros(96, 1024, 1024, device="cuda")
print("keepalive up", flush=True)
while True:
    x.mul_(1.0000001)
    torch.cuda.synchronize()
    time.sleep(20)
