import os
import cv2
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from mobilenetv1_face_landmark import MobileNetV1
from dataloader import CustomDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path = 'F:/07_Data/300W-LP/only_face'
anno_name = 'trntest'
dataset_type = 'imgs'
num_landmark = 136
model_save_path = 'E:/models'

def train(data_loader, model, criterion, optimizer, device, scheduler):
    model.train(True)
    running_loss = 0.0

    for i, data in enumerate(data_loader):
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        y_preds = model(images)

        loss = criterion(y_preds, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step(running_loss / len(data_loader))

    return running_loss / len(data_loader)

def main():
    # t_img = cv2.imread(f'{root_path}/imgs/AFW_1051618982_1_0.jpg')

    lr = 0.001

    models = [MobileNetV1]
    model_name = ['mobilev1']
    for init_model in models:
        img_size = [(224, 224, 3), (224, 224, 1), (256, 256, 3), (256, 256, 1)]

        m_name = model_name[0]
        print(f'Train Model {m_name}')
        for i_size in img_size:
            print(f'Train Image size {i_size}')
            dataset = CustomDataset(root_path, anno_name, dataset_type, i_size)
            print(f"Train dataset size {len(dataset)}")
            train_loader = DataLoader(dataset,
                                      batch_size=8,
                                      num_workers=0,
                                      shuffle=True)

            model = init_model(i_size[0], i_size[-1], num_landmark)
            model.to(DEVICE)

            criterion = torch.nn.MSELoss().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            pre_train = False
            model_path = 'E:/models'
            if pre_train:
                saved_state = torch.load(f'{model_path}/f_landmark-tcdcn-52-0.0004.pth')
                model.load_state_dict(saved_state['model_state_dict'])
                optimizer.load_state_dict(saved_state['optimizer_state_dict'])
                init_epoch = saved_state['Epoch']
                min_loss = saved_state['loss']
            else:
                init_epoch = 0
                min_loss = 1

            epochs = 10000
            print(f'min loss : {min_loss}')
            for epoch in range(init_epoch, epochs):
                print(f'{epoch} epoch start! : {datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")}')

                loss = train(train_loader, model, criterion, optimizer, DEVICE, scheduler)
                print(f"    Average Loss : {loss:.6f}")

                if min_loss > loss:
                    min_loss = loss
                    state = {
                        'Epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }
                    model_path = os.path.join(model_save_path, f'f_landmark-{m_name}-{i_size}.pth')
                    torch.save(state, model_path)
                    print(f'Saved model_{m_name} _ [loss : {loss:.6f}, save_path : {model_path}\n')

                if loss < 0.000001:
                    break

                # draw_img = t_img.copy()
                # t_img = cv2.resize(t_img, (224, 224))
                # x = torch.from_numpy(t_img.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).cuda()
                # # model.eval()
                # y_pred = model(x)
                #
                # for i in range(0, num_landmark, 2):
                #     draw_img = cv2.circle(draw_img, (int(y_pred[0][i].item()*224), int(y_pred[0][i+1].item()*224)), 2, (20, 0, 255), -1)
                # cv2.imwrite(f'{root_path}/test/{epoch}.jpg', draw_img)

if __name__=='__main__':
    main()