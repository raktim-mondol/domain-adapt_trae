import torch


def update_ema(student, teacher, decay=0.99):
    with torch.no_grad():
        for ts, tt in zip(student.parameters(), teacher.parameters()):
            tt.data.mul_(decay).add_(ts.data, alpha=1.0 - decay)

