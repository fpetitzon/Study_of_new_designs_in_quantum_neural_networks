

import numpy as np
from pandas import read_csv, DataFrame, concat
import matplotlib.pyplot as plt
from classes.QnnTorchConnector import *
from sklearn.model_selection import train_test_split
from classes.CsvDataset import *
from classes.QCircuits import *
import torch as T
from qiskit.algorithms.optimizers import SPSA, SLSQP, L_BFGS_B, COBYLA


def train_with_qiskit_optimizer(model, train_loader, test_dl=None, optimizer=None, loss_obj=T.nn.BCELoss()):
    loss_list_train = []
    loss_list_val = []
    acc_list_val = []

    for batch_idx, (data, target) in enumerate(train_loader):

        objective_function = lambda params: model.cost_function(data, target, loss_obj, params)

        params, loss, _ = optimizer.optimize(len(model.circuit.parameters) - model.circuit_length,
                                             objective_function,
                                             initial_point=model.params)
        model.weight = params
        loss_list_train.append(loss)

        if test_dl is not None:
            val_loss, val_acc = eval_on_dataloader(model, test_dl, loss_obj)
            acc_list_val.append(val_acc)
            loss_list_val.append(val_loss)

    print("end of run - train loss array is:", loss_list_train)
    if test_dl is not None:
        print("end of run - val loss array is:", loss_list_val)
        print("end of run - val accuracy is:", acc_list_val)

    return loss_list_val, loss_list_train, acc_list_val


def train_on_dl(model, train_loader, test_dl=None, epochs=100, optimizer=None,
                verbose=0, loss_obj=T.nn.BCELoss()):
    loss_list_val = []
    loss_list_train = []
    acc_list_val = []

    if optimizer is None:
        optimizer = T.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        total_loss = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.type(T.DoubleTensor)

            # Forward pass
            output = model(data)
            target = T.reshape(target, [1, -1])
            output = T.reshape(output, [1, -1])
            # print(output)
            # print(target)

            # Calculating loss
            loss = loss_obj(output, target)
            # print("loss:", loss)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        epoch_train_loss = sum(total_loss) / len(total_loss)
        loss_list_train.append(epoch_train_loss)

        if verbose == 1:
            print("training loss for epoch ", epoch, " is :", loss_list_val[-1])
        if test_dl is not None:
            model.eval()
            val_loss, val_accuracy = eval_on_dataloader(model, test_dl, loss_obj)
            loss_list_val.append(val_loss)
            acc_list_val.append(val_accuracy)

    print("end of run - train loss array is:", loss_list_train)
    print("end of run - val loss array is:", loss_list_val)
    if val_accuracy is not None:
        print("end of run - val accuracy is:", acc_list_val)

    return loss_list_val, loss_list_train, acc_list_val


def eval_on_dataloader(model, data_loader, loss_obj=T.nn.BCELoss()):  # TODO inefficient as it predicts twice...
    loss = compute_loss(model, data_loader, loss_obj)
    acc = accuracy(model, data_loader)
    return loss, acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)


def compute_loss(model, dataloader, loss_obj=T.nn.BCELoss()):
    total_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if isinstance(model, Qnn):
            output = T.tensor(model.predict(inputs))
        else:
            output = model(inputs)
        target = T.reshape(targets, [1, -1])
        output = T.reshape(output, [1, -1])
        total_loss += loss_obj(output, target).item() / len(dataloader)
    return total_loss


def accuracy(model, dataloader):
    acc = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if isinstance(model, Qnn):
            output = T.tensor(model.predict(inputs))
        else:
            output = model(inputs)
        output = T.reshape(output, [1, -1])
        targets = T.reshape(targets, [1, -1])
        acc += T.sum(T.round(output) == T.round(targets)).numpy() / len(dataloader.dataset)
    return acc


def load_loss_and_add_to_plot(path, file, color, epochs=100, label=None):
    loss = np.load(path + file, allow_pickle=True)  # [0]
    if label is None:
        label = file[:-4]
    plt.plot(range(epochs), loss, label=label, color=color)


def plot_two_features_classification(x1, x2, y, filename_for_saving=None):
    x_min, x_max = x1.min() - 0.5, x1.max() + 0.5
    y_min, y_max = x2.min() - 0.5, x2.max() + 0.5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(x1, x2, c=np.round(y), cmap=plt.cm.Set1, edgecolor="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    if filename_for_saving is not None:
        plt.savefig(filename_for_saving)  # example 'GeneratedDataSet2.png'
    plt.show()


def create_train_test_dataloader(dataset, batch_size=100, test_size=0.2, random_state=8):

    train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)
    train_dl = T.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    test_dl = T.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


def create_qnn_and_fit_and_store_result(save_path, circuit_type, dataset, n, epochs,
                                        batch_size=10, loss_obj=T.nn.BCELoss(), test_size=0.2, lrn=0.05):

    print("--------create model-----------")

    circuit1 = CircuitFactory().provide(circuit_type, n)
    qnn1 = QnnTorchConnector(Qnn(circuit1))

    print("param count is :", count_parameters(qnn1))

    train_dl, test_dl = create_train_test_dataloader(dataset, batch_size=batch_size, test_size=test_size)

    loss_array, _, acc_array = train_on_dl(qnn1, train_dl, test_dl, epochs=epochs,
                                           loss_obj=loss_obj, optimizer=T.optim.Adam(qnn1.parameters(), lr=lrn))

    np.save(save_path + circuit_type + '_bs' + str(batch_size) + '_loss.npy', loss_array)
    np.save(save_path + circuit_type + '_bs' + str(batch_size) + '_accuracy.npy', acc_array)
    print("--------end fitting-----------\n")

    return qnn1


def create_hybridqnn_and_fit_and_store_result(save_path, circuit_type, dataset, n, epochs,
                                              batch_size=10, loss_obj=T.nn.BCELoss(), test_size=0.2, lrn=0.05):

    print("--------create model-----------")

    circuit1 = CircuitFactory().provide(circuit_type, n)
    qnn1 = HybridClassificationNet(QnnTorchConnector(Qnn(circuit1, post_process="probabilities_per_qubit")),
                                   T.nn.Linear(n, 1),
                                   binary_classification=True if isinstance(loss_obj, T.nn.BCELoss) else False)

    print("param count is :", count_parameters(qnn1))
    train_dl, test_dl = create_train_test_dataloader(dataset, batch_size=batch_size, test_size=test_size)

    loss_array, _, acc_array = train_on_dl(qnn1, train_dl, test_dl, epochs=epochs,
                                           loss_obj=loss_obj, optimizer=T.optim.Adam(qnn1.parameters(), lr=lrn))

    np.save(save_path + "_hybrid_" + circuit_type + '_bs' + str(batch_size) + '_loss.npy', loss_array)
    np.save(save_path + "_hybrid_" + circuit_type + '_bs' + str(batch_size) + '_accuracy.npy', acc_array)
    print("--------end fitting-----------\n")

    return qnn1


def create_cnn_and_fit_and_store_result(save_path, dataset, n, epochs,
                                        batch_size=10, loss_obj=T.nn.BCELoss(), test_size=0.2, lrn=0.05):

    print("--------create model-----------")
    cnet = ClassicalNet(n, extra_layer=True, neurons=n,
                        binary_classification=True if isinstance(loss_obj, T.nn.BCELoss) else False)
    print("param count is :", count_parameters(cnet))

    train_dl, test_dl = create_train_test_dataloader(dataset, batch_size=batch_size, test_size=test_size)

    loss_array, _, acc_array = train_on_dl(cnet, train_dl, test_dl, epochs=epochs,
                                           loss_obj=loss_obj, optimizer=T.optim.Adam(cnet.parameters(), lr=lrn))

    np.save(save_path + "_cnn_extra_" + '_bs' + str(batch_size) + '_loss.npy', loss_array)
    np.save(save_path + "_cnn_extra_" + '_bs' + str(batch_size) + '_accuracy.npy', acc_array)
    print("--------end fitting-----------\n")

    print("--------create model (without extra layer) -----------")

    cnet2 = ClassicalNet(n, extra_layer=False, neurons=n,
                        binary_classification=True if isinstance(loss_obj, T.nn.BCELoss) else False)
    print("param count is :", count_parameters(cnet2))

    loss_array, _, acc_array = train_on_dl(cnet2, train_dl, test_dl, epochs=epochs,
                                           loss_obj=loss_obj, optimizer=T.optim.Adam(cnet2.parameters(), lr=lrn))

    np.save(save_path + "_cnn_no_extra_" + '_bs' + str(batch_size) + '_loss.npy', loss_array)
    np.save(save_path + "_cnn_no_extra_" + '_bs' + str(batch_size) + '_accuracy.npy', acc_array)
    print("--------end fitting-----------\n")

    return cnet, cnet2
