import torch
import numpy as np
import cv2
from models.STR_transformer import STR_Transformer

torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/classInd.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = STR_Transformer(num_classes=7, at_type="distance_attention", lstm_channel=8)
    checkpoint = torch.load(
        'D:\yangmeng_workspace\ym-action-recognize-v2\exp\experiments_myaction_8_STR_Transformer\STR_best.pth',
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    video = r'D:\DATASET\myaction\ConstructionSafety\wear_helmet/0_wear_helmet_0.avi'
    cap = cv2.VideoCapture(video)
    retaining = True

    # out
    fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (
    #     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # )
    size = (400, 300)
    fourcc = cv2.VideoWriter_fourcc("M", "P", "4", "2")
    outVideo = cv2.VideoWriter("demo.avi", fourcc, fps, size)  # change fps here

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = cv2.resize(frame, (224, 224))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 8:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            scores, preds = probs.topk(3, 1, True, True)
            # scores = torch.nn.Softmax(dim=1)(outputs)
            score = scores.detach().cpu().numpy()
            pred = preds.detach().cpu().numpy()

            frame = cv2.resize(frame, dsize=size)

            cv2.putText(frame, class_names[pred[0][0]].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(frame, "prob: %.4f" % outputs[0][pred[0][0]], (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

            cv2.putText(frame, class_names[pred[0][1]].split(' ')[-1].strip(), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            cv2.putText(frame, "prob: %.4f" % outputs[0][pred[0][1]], (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            clip.pop(0)

        cv2.imshow('result', frame)
        outVideo.write(frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
