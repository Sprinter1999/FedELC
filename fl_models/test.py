import torch.nn.functional as F  
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  
  
def test_img(net_g, data_loader, args):  
    net_g.eval()  
    test_loss = 0  
    n_total = len(data_loader.dataset)  
    y_true = []  
    y_pred = []  
  
    with torch.no_grad():  
        for idx, (data, target) in enumerate(data_loader):  
            if args.gpu != -1:  
                data, target = data.to(args.device), target.to(args.device)  
  
            log_probs, __ = net_g(data)  
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()  
  
            y_pred_batch = log_probs.data.max(1, keepdim=True)[1]  
            y_true_batch = target.data.view_as(y_pred_batch)  
  
            y_true.extend(y_true_batch.cpu().numpy())  
            y_pred.extend(y_pred_batch.cpu().numpy())  
  
    test_loss /= n_total  
    accuracy = accuracy_score(y_true, y_pred) * 100.0  
    precision = precision_score(y_true, y_pred, average='macro')  
    recall = recall_score(y_true, y_pred, average='macro')  
    f1 = f1_score(y_true, y_pred, average='macro')  
  
    # if args.verbose:  
    print('Test set: Average loss: {:.4f}'.format(test_loss))  
    print('Accuracy: {:.2f}%'.format(accuracy))  
    print('Precision: {:.4f}'.format(precision))  
    print('Recall: {:.4f}'.format(recall))  
    print('F1 Score: {:.4f}'.format(f1))  
  
    return accuracy, test_loss, precision, recall, f1  

# 提取中间特征嵌入  
def get_feature_embedding(model, loader):  
    embeddings = []  
    model.eval()  
    with torch.no_grad():  
        for images, _ in loader:  
            # 获取中间特征嵌入
            images = images.cuda()
            out, features = model(images)  
            embeddings.append(features.squeeze().detach().cpu().numpy())  
    embeddings = torch.tensor(embeddings)  
    return embeddings  



def test_img_svd(net_g, data_loader, args):  
    net_g.eval()  
    test_loss = 0  
    n_total = len(data_loader.dataset)  
    y_true = []  
    y_pred = []  
  
    with torch.no_grad():  
        for idx, (data, target) in enumerate(data_loader):  
            if args.gpu != -1:  
                data, target = data.to(args.device), target.to(args.device)  
  
            log_probs, __ = net_g(data)  
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()  
  
            y_pred_batch = log_probs.data.max(1, keepdim=True)[1]  
            y_true_batch = target.data.view_as(y_pred_batch)  
  
            y_true.extend(y_true_batch.cpu().numpy())  
            y_pred.extend(y_pred_batch.cpu().numpy())  
  
    test_loss /= n_total  
    accuracy = accuracy_score(y_true, y_pred) * 100.0  
    precision = precision_score(y_true, y_pred, average='macro')  
    recall = recall_score(y_true, y_pred, average='macro')  
    f1 = f1_score(y_true, y_pred, average='macro')  
  
    # if args.verbose:  
    print('Test set: Average loss: {:.4f}'.format(test_loss))  
    print('Accuracy: {:.2f}%'.format(accuracy))  
    print('Precision: {:.4f}'.format(precision))  
    print('Recall: {:.4f}'.format(recall))  
    print('F1 Score: {:.4f}'.format(f1))  
  
    return accuracy, test_loss, precision, recall, f1  