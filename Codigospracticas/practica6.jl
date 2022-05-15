using ScikitLearn
using Random
using Random:seed!
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
include("practica5.jl")
function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict,
    inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));
    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion
    # one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);
    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;
    # Creamos los indices de crossvalidation
    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1 = Array{Float64,1}(undef, numFolds);
    # Para cada fold, entrenamos
    for numFold in 1:numFolds
        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)
            # Dividimos los datos en entrenamiento y test
            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold];
            testTargets = targets[crossValidationIndices.==numFold];
            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"],
                degree=modelHyperparameters["kernelDegree"],
                gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);

            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;
        # Entrenamos el modelo con el conjunto de entrenamiento
        model = fit!(model, trainingInputs, trainingTargets);
        # Pasamos el conjunto de test
        testOutputs = predict(model, testInputs);
        testOutputs=oneHotEncoding(testOutputs)
        testTargets=oneHotEncoding(testTargets)

        # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior

        (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs,testTargets);
        else
            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);
            # Dividimos los datos en entrenamiento y test
            trainingInputs = inputs[crossValidationIndices.!=numFold,:];
            testInputs = inputs[crossValidationIndices.==numFold,:];
            trainingTargets = targets[crossValidationIndices.!=numFold,:];
            testTargets = targets[crossValidationIndices.==numFold,:];
            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            # se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef,
            modelHyperparameters["numExecutions"]);
            testF1EachRepetition = Array{Float64,1}(undef,
            modelHyperparameters["numExecutions"]);
            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"]>0
                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    # dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    # Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) =
                    holdOut(size(trainingInputs,1),
                    modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                    trainingInputs[trainingIndices,:],
                    trainingTargets[trainingIndices,:],
                    trainingInputs[validationIndices,:],
                    trainingTargets[validationIndices,:],
                    testInputs, testTargets;
                    maxEpochs=modelHyperparameters["maxEpochs"],
                    learningRate=modelHyperparameters["learningRate"],
                    maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
                else
                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    # teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                    (trainingInputs, trainingTargets),
                    (testInputs, testTargets);
                    maxEpochs=modelHyperparameters["maxEpochs"],
                    learningRate=modelHyperparameters["learningRate"]);
                end;
                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _,
                testF1EachRepetition[numTraining], _) =
                confusionMatrix(collect(ann(testInputs')'), testTargets);
            end;
            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1 = mean(testF1EachRepetition);
        end;
        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold] = F1;
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy:
            ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
     end; # for numFold in 1:numFolds
    println(modelType, ": Average test accuracy on a ", numFolds, "-fold
    crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ",

    100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold
    crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ",
    100*std(testF1));
    return (mean(testAccuracies), std(testAccuracies), mean(testF1),
    std(testF1));
end;

# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);
numFolds = 10;
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento
# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;
# Parametros del arbol de decision
maxDepth = 4;
# Parapetros de kNN
numNeighbors = 3;
# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = dataset[:,5];
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, numFolds);
# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);
# Entrenamos los arboles de decision
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs,
targets, numFolds);
# Entrenamos los kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs,
targets, numFolds);
