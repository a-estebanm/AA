include("script4.2.jl")
include("script3.jl")
using Random
using Random:seed!
function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;


function crossvalidation(targets::AbstractArray{Bool,2},k::Int64)
    indices = Array{Int64,1}(undef, size(targets,1))
    for numClass in 1:size(targets,2)
        indices[1+((numClass-1)*sum(targets[:, numClass])):numClass*sum(targets[:, numClass])].=crossvalidation(sum(targets[:, numClass]),k)
    end
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1},k::Int64)
    crossvalidation(oneHotEncoding(targets),k)
end
# -------------------------------------------------------------------------
# Código de prueba:
# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
numFolds = 10;
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion.
#Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el 
#conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA
#para cada fold por el hecho de ser no determinístico el entrenamiento
# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = convert(AbstractArray{Any,1}, dataset[:,5])
targets = oneHotEncoding(targets);
numClasses = size(targets,2);
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
@assert(numClasses>2);
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);
# Creamos los indices de crossvalidation
crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
# Creamos los vectores para las metricas que se vayan a usar
# En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
testAccuracies = Array{Float64,1}(undef, numFolds);
testF1 = Array{Float64,1}(undef, numFolds);
# Para cada fold, entrenamos
for numFold in 1:numFolds
    # Dividimos los datos en entrenamiento y test
    local trainingInputs, testInputs, trainingTargets, testTargets;
    trainingInputs = inputs[crossValidationIndices.!=numFold,:];
    testInputs = inputs[crossValidationIndices.==numFold,:];
    trainingTargets = targets[crossValidationIndices.!=numFold,:];
    testTargets = targets[crossValidationIndices.==numFold,:];
    # En el caso de entrenar una RNA, este proceso es no determinístico, por lo que es necesario repetirlo para cada fold
    # Para ello, se crean vectores adicionales para almacenar las metricas para cada entrenamiento
    testAccuraciesEachRepetition = Array{Float64,1}(undef,
    numRepetitionsAANTraining);
    testF1EachRepetition = Array{Float64,1}(undef,
    numRepetitionsAANTraining);
    for numTraining in 1:numRepetitionsAANTraining
        if validationRatio>0
            # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
            # dividimos el conjunto de entrenamiento en entrenamiento+validacion
            # Para ello, hacemos un hold out
            local trainingIndices, validationIndices;
            (trainingIndices, validationIndices) =
                holdOut(size(trainingInputs,1),
                validationRatio*size(trainingInputs,1)/size(inputs,1));
            # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
            # Entrenamos la RNA
            local ann;
            ann, = trainClassANN(topology,
                trainingInputs[trainingIndices,:],
                trainingTargets[trainingIndices,:],
                trainingInputs[validationIndices,:],
                trainingTargets[validationIndices,:],

                testInputs, testTargets;
                maxEpochs=numMaxEpochs, learningRate=learningRate,
                maxEpochsVal=maxEpochsVal);
        else
            # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test
            local ann;
            ann, = trainClassANN(topology,
            (trainingInputs, trainingTargets),
            (testInputs, testTargets);
            maxEpochs=numMaxEpochs, learningRate=learningRate);
        end;
        # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
        (acc, _, _, _, _, _, F1, _) = confusionMatrix(collect(ann(testInputs')'), testTargets);
        # Almacenamos las metricas de este entrenamiento
        testAccuraciesEachRepetition[numTraining] = acc;
        testF1EachRepetition[numTraining] = F1;
    end;
    # Almacenamos las 2 metricas que usamos en este problema
    testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
    testF1[numFold] = mean(testF1EachRepetition);
    println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ",
        100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
end;
println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ",
    100*mean(testAccuracies), ", with a standard deviation of ",
    100*std(testAccuracies));
println("Average test F1 on a ", numFolds, "-fold crossvalidation: ",
    100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));#