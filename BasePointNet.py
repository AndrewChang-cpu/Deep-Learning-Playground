import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from torchvision.transforms import ToTensorfrom
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
import copy
from collections import Counter


try:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
except:
    device = 'cpu'


class KDTreeDataset(Dataset):
    """
    Creates a dataset for training points using a KDTree.

    ...

    Attributes
    ----------
    dim_scalars : list of int
        the scales by which x, y, z were scaled down
    data : numpy array of floats
        a 2D array of points
    tree : KDTree
        a KDTree created with all the points

    """

    def __init__(self, pts_file, split=0):
        """
        Constructs necessary attributes and scales x, y, z

        pts_file : str
            a text file with lints of points in this format:
            x, y, z, intensity, return number, number of returns, label
        split : int
            integer representing how the data should be split. 0: no split, 1: process first half of data, 2: process second half of data
        """
        
        points = np.loadtxt(pts_file, delimiter=' ')
        points = np.delete(points, -2, 1)
        points = np.delete(points, -2, 1)

        if split == 1:
            points = points[:len(points)//2]
        elif split == 2:
            points = points[len(points)//2:]

        print('file points shape', points.shape)
        
        self.dim_scalars = []
        for i in range(4):
            if i == 3:
                points[:,i] = points[:,i] / 255
            else:
                dim_min, dim_max = min(points[:,i]), max(points[:,i])
                points[:,i] = (points[:,i] - dim_min) / (dim_max - dim_min)
                self.dim_scalars.append(dim_max - dim_min)
        self.data = points

        self.tree = KDTree(self.data[:, :3])
        
    def __len__(self):
        return len(self.data) // 15000
        
    def __getitem__(self, idx):
        """
        Randomly jitters the queried point by some small value. Then uses a KDTree to return the nearest 25000 points.
        """

        noisy_point = self.data[idx * 15000, :3] + np.random.normal(0, 0.01, 3)[0]
        
        _, indicies = self.tree.query(noisy_point, k=25000)

        xyzirn = self.data[indicies, :-1]  # x, y, z, intensity, ***NOT INCLUDED: return number, number of returns***
        label = self.data[indicies, -1] == 5 # JUST DOES BINARY CLASSIFICATION FOR ONE CLASS

        xyzirn = torch.from_numpy(xyzirn.T).float()
        label = torch.tensor(label).long()
        
        # print(xyzirn.size(), label.size())

        return xyzirn, label


class NormalDataset(Dataset):
    """
    Creates a normal dataset with no KDTree.

    ...

    Attributes
    ----------
    dim_scalars : list of int
        the scales by which x, y, z were scaled down
    data : numpy array of floats
        a 2D array of points

    """
    def __init__(self, pts_file, split=0):
        """
        Constructs necessary attributes and scales x, y, z

        pts_file : str
            a text file with lints of points in this format:
            x, y, z, intensity, return number, number of returns, label
        split : int
            integer representing how the data should be split. 0: no split, 1: process first half of data, 2: process second half of data
        """

        points = np.loadtxt(pts_file, delimiter=' ')
        points = np.delete(points, -2, 1)
        points = np.delete(points, -2, 1)

        if split == 1:
            points = points[:len(points)//2]
        elif split == 2:
            points = points[len(points)//2:]

        print(points.shape)
        self.dim_scalars = []
        for i in range(4):
            if i == 3:
                points[:,i] = points[:,i] / 255
            else:
                dim_min, dim_max = min(points[:,i]), max(points[:,i])
                points[:,i] = (points[:,i] - dim_min) / (dim_max - dim_min)
                self.dim_scalars.append(dim_max - dim_min)
        self.data = points
        
    def __len__(self):
        return len(self.data) // 25000
        
    def __getitem__(self, idx):
        # Return batches of 25000 points
        xyzirn = self.data[idx * 25000: (idx + 1) * 25000, :-1]
        label = self.data[idx * 25000: (idx + 1) * 25000, -1] == 5

        xyzirn = torch.from_numpy(xyzirn.T).float()
        label = torch.tensor(label).long()
        
        return xyzirn, label


def plot_view(training_data, entire_view = True):
    # Assuming pc is your point cloud data, in shape (N, 3)
    pc_num = 2

    if entire_view:
        pc = training_data.data[:, :3]
        labels = training_data.data[:, -1] == 5
    else:
        pc, labels = training_data[pc_num]
        pc = pc.T.numpy()[:, :3]
        labels = labels.numpy()

    pc = copy.deepcopy(pc)
    for i in range(3):
        pc[:, i] *= training_data.dim_scalars[i]

    print(Counter(labels), len(labels))

    # Define colors for each label
    color_map = {0: [0.5, 0.5, 0.5],  # Gray color for label 0
                1: [1.0, 0.0, 0.0]}  # Red color for label 1

    # Map each label to a color
    colors = np.array([color_map[label] for label in labels])

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


def plot_batch(training_data, batch_data, batch_labels):
    pc, batch_labels = batch_data.transpose(1,2).reshape(-1, 4).numpy()[:, :3], batch_labels.view(-1).numpy()
    # print(batch_labels.shape)
    pc = copy.deepcopy(pc)
    for i in range(3):
        pc[:, i] *= training_data.dim_scalars[i]

    print(Counter(list(batch_labels)), len(batch_labels))

    # Define colors for each label
    color_map = {0: [0.5, 0.5, 0.5],  # Gray color for label 0
                1: [1.0, 0.0, 0.0]}  # Red color for label 1

    # Map each label to a color
    colors = np.array([color_map[label] for label in batch_labels])

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


class STNkd(nn.Module):
    """
    T-net implementation to make sure batches are standardized despite transformations
    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.mlp1 = nn.Sequential(torch.nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64), nn.GELU())
        self.mlp2 = nn.Sequential(torch.nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.GELU())
        self.mlp3 = nn.Sequential(torch.nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.GELU())
        self.mlp4 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU())
        self.mlp5 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU())
        self.fc = nn.Linear(256, k*k)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.fc(x)

        iden = torch.eye(self.k, requires_grad=True).repeat(batchsize,1,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.k, self.k) + iden

        return x


class PointNet(nn.Module):
    """
    PointNet implementation that takes in a tensor of size [batchsize, in_channels, points] 
    and returns a tensor of size [batchsize, points, num_classes]
    """
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()

        self.stn1 = STNkd(k=in_channels)
        self.mlp1 = nn.Sequential(nn.Conv1d(in_channels, 64, kernel_size=1), nn.BatchNorm1d(64), nn.GELU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1), nn.BatchNorm1d(64), nn.GELU())
        
        self.stn2 = STNkd(k=64)
        self.mlp3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1), nn.BatchNorm1d(64), nn.GELU())
        self.mlp4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1), nn.BatchNorm1d(128), nn.GELU())
        self.mlp5 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1), nn.BatchNorm1d(1024), nn.GELU())
        
        self.mlp6 = nn.Sequential(nn.Linear(1088, 512), nn.LayerNorm(512), nn.GELU())
        self.mlp7 = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU())
        self.mlp8 = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.fc = nn.Linear(128, num_classes)

        self.debug = nn.Conv1d(in_channels, num_classes, 1)

    def forward(self, x):
        n_pts = x.size()[2]

        trans6x6 = self.stn1(x) 
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans6x6)
        x = x.transpose(2, 1)

        x = self.mlp1(x)
        # print(1, x.size())
        x = self.mlp2(x)
        # print(2, x.size())
        
        
        trans64x64 = self.stn2(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans64x64)
        local_features = x.transpose(2, 1)
        
        x = self.mlp3(local_features)
        # print(3, x.size())
        x = self.mlp4(x)
        # print(4, x.size())
        x = self.mlp5(x)
        # print(5, x.size())
        x = torch.max(x, 2)[0]
        # print(6, x.size())

        global_features = x.unsqueeze(2).repeat(1, 1, n_pts)
        # print(7, global_features.size())
        x = torch.cat([local_features, global_features], 1)
        # print(8, x.size())

        x = x.transpose(2, 1)
        x = self.mlp6(x)
        # print(9, x.size())
        x = self.mlp7(x)
        # print(10, x.size())
        x = self.mlp8(x)
        # print(11, x.size())
        x = self.fc(x)
        # print(12, x.size())

        # print(13, x.size())
        x = F.log_softmax(x, dim=-1)
        # print(14, x.size())

        return x, trans64x64


def feature_transform_regularizer(trans):
    d = trans.size(1)
    I = torch.eye(d).unsqueeze(0).to(device)
    loss = torch.linalg.norm(I - torch.bmm(trans, trans.transpose(2,1)), dim=(1,2))
    loss = torch.mean(loss)
    return loss


def train_pointnet(classifier, train_dataloader, validation_dataloader, num_classes, epochs=120):
    writer = SummaryWriter()

    optimizer = optim.Adam(classifier.parameters(), lr=0.0005, betas=(0.9, 0.999))

    for epoch in range(epochs):
        classifier.train()
        train_loss, train_f1, train_acc = 0.0, 0.0, 0.0
        predictions, labels = np.array([]), np.array([])
        for i, data in enumerate(train_dataloader, 1):
            points, target = data
            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()
            pred, trans_feat = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1).squeeze()
            print(f'Iteration {i}', pred.size(), target.size())

            loss = F.nll_loss(pred, target)
            loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            train_loss += float(loss) # JUST USING LOSS ACCUMULATES HISTORY

            predictions = np.append(predictions, pred.max(1)[1].cpu())
            labels = np.append(labels, target.cpu())

        train_f1 = f1_score(predictions, labels)
        train_acc = sum(predictions == labels)/float(len(labels))
        train_loss /= len(train_dataloader)


        classifier.eval()
        valid_loss, valid_f1, valid_acc = 0.0, 0.0, 0.0
        predictions, labels = np.array([]), np.array([])
        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                points, target = data
                points, target = points.to(device), target.to(device)
                pred,  trans_feat = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1).squeeze()

                loss = F.nll_loss(pred, target)
                loss += feature_transform_regularizer(trans_feat) * 0.001
                valid_loss += float(loss)

                predictions = np.append(predictions, pred.max(1)[1].cpu())
                labels = np.append(labels, target.cpu())

        valid_f1 = f1_score(predictions, labels)
        valid_acc = sum(predictions == labels)/float(len(labels))
        valid_loss /= len(validation_dataloader)


        writer.add_scalars('losses', {'training':train_loss, 'validation':valid_loss}, global_step=epoch)
        writer.add_scalars('f1 scores', {'training':train_f1, 'validation':valid_f1}, global_step=epoch)

        print(f'[{epoch}] train loss: {train_loss} accuracy: {train_acc} f1 score: {train_f1}')
        print(f'[{epoch}] validation loss: {valid_loss} accuracy: {valid_acc} f1 score: {valid_f1}')
        print()

    writer.flush()
    writer.close()


def test_PointNet(pointnet, test_dataloader):
    global outputs, points, x, y
    pointnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (points, labels) in enumerate(test_dataloader):
            points, labels = points.to(device), labels.to(device)
            outputs, _ = pointnet(points)
            _, predicted = torch.max(outputs.data, 2)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            x = labels.view(-1).to('cpu').numpy()
            y = predicted.view(-1).to('cpu').numpy()
            f1 = f1_score(x, y)
            print('F1 score: ', f1)
            # plot_batch(points, predicted)
            # plot_batch(points, labels)


def main():
    print('Started program')

    num_classes = 2 # dependent on KDTreeDataset and NormalDataset implementation

    # Read data into Datasets
    print('Started creating dataloaders')
    data_folder = os.path.join(os.getcwd(), r"data\Point Cloud")
    training_data = KDTreeDataset(os.path.join(data_folder, r"Traininig.pts"))
    validation_data = NormalDataset(os.path.join(data_folder, r"Testing.pts"), split=1)
    testing_data = NormalDataset(os.path.join(data_folder, r"Testing.pts"), split=2)

    # Load Datasets into DataLoader
    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=5, shuffle=False)
    test_dataloader = DataLoader(testing_data, batch_size=5, shuffle=False)
    print('Finished creating dataloaders')

    # Print device for training
    print(f"Using {device} device")

    # Initialize, train, and test PointNet model
    print('Started training model')
    classifier = PointNet(in_channels=4, num_classes=2).to(device)
    train_pointnet(classifier, train_dataloader, validation_dataloader, num_classes, epochs=1)
    test_PointNet(classifier, test_dataloader)
    print('Finished training model')

    print('Finished program')


if __name__ == "__main__":
    main()