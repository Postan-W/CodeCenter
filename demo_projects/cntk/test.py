# seeds_dnn.py
# classify wheat seed variety
#因为没有数据，这里不训练评估
import numpy as np
import cntk as C
def create_reader(path, is_training, input_dim, output_dim):
    strm_x = C.io.StreamDef(field='properties',
    shape=input_dim, is_sparse=False)
    strm_y = C.io.StreamDef(field='variety',
    shape=output_dim, is_sparse=False)
    streams = C.io.StreamDefs(x_src=strm_x,y_src=strm_y)
    deserial = C.io.CTFDeserializer(path, streams)
    sweeps = C.io.INFINITELY_REPEAT if is_training else 1
    mb_source = C.io.MinibatchSource(deserial,
    randomize=is_training, max_sweeps=sweeps)
    return mb_source
def main():
    print("\nBegin wheat seed classification demo\n")
    print("Using CNTK verson = " + str(C.__version__) + "\n")
    input_dim = 128
    hidden_dim = 500
    output_dim = 10
    train_file = ".\\Data\\seeds_train_data.txt"
    test_file = ".\\Data\\seeds_test_data.txt"
    # 1. create network and model
    X = C.ops.input_variable(input_dim, np.float32)
    Y = C.ops.input_variable(output_dim, np.float32)
    print("Creating a 7-(4-4-4)-3 tanh softmax NN for seed data")
    with C.layers.default_options(init= \
        C.initializer.normal(scale=0.1, seed=2)):
        h1 = C.layers.Dense(hidden_dim, activation=C.ops.tanh,
        name='hidLayer1')(X)
        h2 = C.layers.Dense(hidden_dim, activation=C.ops.tanh,
        name='hidLayer2')(h1)
        h3 = C.layers.Dense(hidden_dim, activation=C.ops.tanh,
        name='hidLayer3')(h2)
        oLayer = C.layers.Dense(output_dim, activation=None,
        name='outLayer')(h3)
    nnet = oLayer
    model = C.softmax(nnet)
    # 2. create learner and trainer
    print("Creating a cross entropy, SGD with LR=0.01,batch=10 Trainer \n")
    tr_loss = C.cross_entropy_with_softmax(nnet, Y)
    tr_clas = C.classification_error(nnet, Y)
    learn_rate = 0.01
    learner = C.sgd(nnet.parameters, learn_rate)
    trainer = C.Trainer(nnet, (tr_loss, tr_clas), [learner])
    max_iter = 50# maximum training iterations
    batch_size = 10# mini-batch size
    # 3. create data reader
    my_input_map = {X : np.random.randn(100,input_dim),Y : np.random.randint(size=100,low=0,high=10)}
    # 4. train
    print("Starting training \n")
    # for i in range(0, max_iter):
    #     trainer.train_minibatch(my_input_map)
    #     if i % 1000 == 0:
    #         mcee = trainer.previous_minibatch_loss_average
    #         pmea = trainer.previous_minibatch_evaluation_average
    #         macc = (1.0 - pmea) * 100
    #         print("batch %6d: mean loss = %0.4f,mean accuracy = %0.2f%% " % (i,mcee, macc))
    # print("\nTraining complete")
    # # 5. evaluate model on the test data
    # print("\nEvaluating test data \n")
    # rdr = create_reader(test_file, False, input_dim, output_dim)
    # my_input_map = {X : rdr.streams.x_src,Y : rdr.streams.y_src}
    # numTest = 60
    # allTest = rdr.next_minibatch(numTest, input_map=my_input_map)
    # acc = (1.0 - trainer.test_minibatch(allTest)) * 100
    # print("Classification accuracy on the 60 test items = %0.2f%%" % acc)
    # (could save model here)
    # 6. use trained model to make prediction
    np.set_printoptions(precision = 4)
    unknown = np.random.randn(1,input_dim)
    print("\nPredicting variety for (non-normalized) seed features: ")
    print(unknown[0])
    raw_out = model.eval({X: unknown})
    print(raw_out)
    model.save("./cntkmodel.model", format=C.ModelFormat.CNTKv2)
# main()
if __name__ == "__main__":
    main()
