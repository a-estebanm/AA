using FileIO;
using DelimitedFiles;
using Statistics
using Flux
using Flux.Losses
using Random
using Random:seed!
using JLD2
using Images
using ScikitLearn
using Augmentor
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

include("processing.jl")

# Funcion para realizar la codificacion, recibe el vector de caracteristicas (uno por patron), y las clases
function oneHotEncoding(feature::Array{Any,1}, classes::Array{Any,1})
    # Primero se comprueba que todos los elementos del vector esten en el vector de clases (linea adaptada del final de la practica 4)
    @assert(all([in(value, classes) for value in feature]))
    numClasses = length(classes)
    @assert(numClasses > 1)
    if (numClasses == 2)
        # Si solo hay dos clases, se devuelve una matriz con una columna
        oneHot = Array{Bool,2}(undef, size(feature, 1), 1)
        oneHot[:, 1] .= (feature .== classes[1])
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase
        oneHot = Array{Bool,2}(undef, size(feature, 1), numClasses)
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
    end
    return oneHot
end;
# Esta funcion es similar a la anterior, pero si no es especifican las clases, se toman de la propia variable
oneHotEncoding(feature::Array{Any,1}) =
    oneHotEncoding(feature::Array{Any,1}, unique(feature));
# Sobrecargamos la funcion oneHotEncoding por si acaso pasan un vector de valores booleanos
#  En este caso, el propio vector ya está codificado
oneHotEncoding(feature::Array{Bool,1}) = feature;
# Cuando se llame a la funcion oneHotEncoding, según el tipo del argumento pasado, Julia realizará
#  la llamada a la función correspondiente

# -------------------------------------------------------
# Funciones auxiliar que permite transformar una matriz de
#  valores reales con las salidas del clasificador o clasificadores
#  en una matriz de valores booleanos con la clase en la que sera clasificada
function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool = true)
    # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
    (_, indicesMaxEachInstance) = findmax(outputs, dims = dataInRows ? 2 : 1)
    # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
    outputsBoolean = Array{Bool,2}(falses(size(outputs)))
    outputsBoolean[indicesMaxEachInstance] .= true
    # Comprobamos que efectivamente cada patron solo este clasificado en una clase
    @assert(all(sum(outputsBoolean, dims = dataInRows ? 2 : 1) .== 1))
    return outputsBoolean
end;

# -------------------------------------------------------
# Funciones para calcular la precision

accuracy(outputs::Array{Bool,1}, targets::Array{Bool,1}) = mean(outputs .== targets);
function accuracy(outputs::Array{Bool,2}, targets::Array{Bool,2}; dataInRows::Bool = true)
    @assert(all(size(outputs) .== size(targets)))
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets, 2) == 1)
            return accuracy(outputs[:, 1], targets[:, 1])
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims = 2)
            return mean(correctClassifications)
        end
    else
        # Cada patron esta en cada columna
        if (size(targets, 1) == 1)
            return accuracy(outputs[1, :], targets[1, :])
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims = 1)
            return mean(correctClassifications)
        end
    end
end;

accuracy(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64 = 0.5) =
    accuracy(Array{Bool,1}(outputs .>= threshold), targets);
function accuracy(
    outputs::Array{Float64,2},
    targets::Array{Bool,2};
    dataInRows::Bool = true,
)
    @assert(all(size(outputs) .== size(targets)))
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets, 2) == 1)
            return accuracy(outputs[:, 1], targets[:, 1])
        else
            return accuracy(classifyOutputs(outputs; dataInRows = true), targets)
        end
    else
        # Cada patron esta en cada columna
        if (size(targets, 1) == 1)
            return accuracy(outputs[1, :], targets[1, :])
        else
            return accuracy(classifyOutputs(outputs; dataInRows = false), targets)
        end
    end
end;

# Añado estas funciones porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funciones se pueden usar indistintamente matrices de Float32 o Float64
accuracy(outputs::Array{Float32,1}, targets::Array{Bool,1}; threshold::Float64 = 0.5) =
    accuracy(Float64.(outputs), targets; threshold = threshold);
accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2}; dataInRows::Bool = true) =
    accuracy(Float64.(outputs), targets; dataInRows = dataInRows);


# -------------------------------------------------------
# Funciones para crear y entrenar una RNA

function buildClassANN(numInputs::Int64, topology::Array{Int64,1}, numOutputs::Int64)
    ann = Chain()
    numInputsLayer = numInputs
    for numOutputLayers in topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ))
        numInputsLayer = numOutputLayers
    end
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end
    return ann
end;

# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
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
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs);

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);

        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:],
                        trainingInputs[validationIndices,:], trainingTargets[validationIndices,:],
                        testInputs,                          testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs, trainingTargets,
                        testInputs,     testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                println(collect(ann(testInputs')'))
                println(testTargets)
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;

# Para calcular los parametros de normalizacion, segun la forma de normalizar que se desee:
calculateMinMaxNormalizationParameters(dataset::Array{Float64,2}; dataInRows = true) = (
    minimum(dataset, dims = (dataInRows ? 1 : 2)),
    maximum(dataset, dims = (dataInRows ? 1 : 2)),
);

calculateZeroMeanNormalizationParameters(dataset::Array{Float64,2}; dataInRows = true) =
    (mean(dataset, dims = (dataInRows ? 1 : 2)), std(dataset, dims = (dataInRows ? 1 : 2)));

# 4 versiones de la funcion para normalizar entre 0 y 1:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeMinMax!(
    dataset::Array{Float64,2},
    normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
    dataInRows = true,
)
    min = normalizationParameters[1]
    max = normalizationParameters[2]
    dataset .-= min
    dataset ./= (max .- min)
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    if (dataInRows)
        dataset[:, vec(min .== max)] .= 0
    else
        dataset[vec(min .== max), :] .= 0
    end
end;
normalizeMinMax!(dataset::Array{Float64,2}; dataInRows = true) = normalizeMinMax!(
    dataset,
    calculateMinMaxNormalizationParameters(dataset; dataInRows = dataInRows);
    dataInRows = dataInRows,
);
function normalizeMinMax(
    dataset::Array{Float64,2},
    normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
    dataInRows = true,
)
    newDataset = copy(dataset)
    normalizeMinMax!(newDataset, normalizationParameters; dataInRows = dataInRows)
    return newDataset
end;
normalizeMinMax(dataset::Array{Float64,2}; dataInRows = true) = normalizeMinMax(
    dataset,
    calculateMinMaxNormalizationParameters(dataset; dataInRows = dataInRows);
    dataInRows = dataInRows,
);


# 4 versiones similares de la funcion para normalizar de media 0:
#  - Nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - No nos dan los parametros de normalizacion, y se quiere modificar el array de entradas (el nombre de la funcion acaba en '!')
#  - Nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
#  - No nos dan los parametros de normalizacion, y no se quiere modificar el array de entradas (se crea uno nuevo)
function normalizeZeroMean!(
    dataset::Array{Float64,2},
    normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
    dataInRows = true,
)
    avg = normalizationParameters[1]
    stnd = normalizationParameters[2]
    dataset .-= avg
    dataset ./= stnd
    # Si hay algun atributo en el que todos los valores son iguales, se pone a 0
    if (dataInRows)
        dataset[:, vec(stnd .== 0)] .= 0
    else
        dataset[vec(stnd .== 0), :] .= 0
    end
end;
normalizeZeroMean!(dataset::Array{Float64,2}; dataInRows = true) = normalizeZeroMean!(
    dataset,
    calculateZeroMeanNormalizationParameters(dataset; dataInRows = dataInRows);
    dataInRows = dataInRows,
);
function normalizeZeroMean(
    dataset::Array{Float64,2},
    normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
    dataInRows = true,
)
    newDataset = copy(dataset)
    normalizeZeroMean!(newDataset, normalizationParameters; dataInRows = dataInRows)
    return newDataset
end;
normalizeZeroMean(dataset::Array{Float64,2}; dataInRows = true) = normalizeZeroMean(
    dataset,
    calculateZeroMeanNormalizationParameters(dataset; dataInRows = dataInRows);
    dataInRows = dataInRows,
);


using Random

function holdOut(N::Int, P::Float64)
    @assert ((P >= 0.0) & (P <= 1.0))
    indices = randperm(N)
    numTrainingInstances = Int(round(N * (1 - P)))
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end])
end

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert ((Pval >= 0.0) & (Pval <= 1.0))
    @assert ((Ptest >= 0.0) & (Ptest <= 1.0))
    @assert ((Pval + Ptest) <= 1.0)
    # Primero separamos en entrenamiento+validation y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest)
    # Después separamos el conjunto de entrenamiento+validation
    (trainingIndices, validationIndices) = holdOut(
        length(trainingValidationIndices),
        Pval * N / length(trainingValidationIndices),
    )
    return (
        trainingValidationIndices[trainingIndices],
        trainingValidationIndices[validationIndices],
        testIndices,
    )
end;


# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de validacion, y parar el entrenamiento si es necesario
function trainClassANN(
    topology::Array{Int64,1},
    trainingInputs::Array{Float64,2},
    trainingTargets::Array{Bool,2},
    validationInputs::Array{Float64,2},
    validationTargets::Array{Bool,2},
    testInputs::Array{Float64,2},
    testTargets::Array{Bool,2};
    maxEpochs::Int64 = 1000,
    minLoss::Float64 = 0.0,
    learningRate::Float64 = 0.1,
    maxEpochsVal::Int64 = 6,
    showText::Bool = false,
)

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validation como test
    @assert(size(trainingInputs, 1) == size(trainingTargets, 1))
    @assert(size(validationInputs, 1) == size(validationTargets, 1))
    @assert(size(testInputs, 1) == size(testTargets, 1))
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento, validacion y test
    @assert(size(trainingInputs, 2) == size(validationInputs, 2) == size(testInputs, 2))
    @assert(size(trainingTargets, 2) == size(validationTargets, 2) == size(testTargets, 2))
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs, 2), topology, size(trainingTargets, 2))
    # Definimos la funcion de loss
    loss(x, y) =
        (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) :
        Losses.crossentropy(ann(x), y)
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[]
    trainingAccuracies = Float64[]
    validationLosses = Float64[]
    validationAccuracies = Float64[]
    testLosses = Float64[]
    testAccuracies = Float64[]

    # Empezamos en el ciclo 0
    numEpoch = 0

    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets')
        validationLoss = loss(validationInputs', validationTargets')
        testLoss = loss(testInputs', testTargets')
        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs')
        validationOutputs = ann(validationInputs')
        testOutputs = ann(testInputs')
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        #  Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc =
            accuracy(trainingOutputs, Array{Bool,2}(trainingTargets'); dataInRows = false)
        validationAcc = accuracy(
            validationOutputs,
            Array{Bool,2}(validationTargets');
            dataInRows = false,
        )
        testAcc = accuracy(testOutputs, Array{Bool,2}(testTargets'); dataInRows = false)
        #  Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc =
            accuracy(Array{Float64,2}(trainingOutputs'), trainingTargets; dataInRows = true)
        validationAcc = accuracy(
            Array{Float64,2}(validationOutputs'),
            validationTargets;
            dataInRows = true,
        )
        testAcc = accuracy(Array{Float64,2}(testOutputs'), testTargets; dataInRows = true)
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println(
                "Epoch ",
                numEpoch,
                ": Training loss: ",
                trainingLoss,
                ", accuracy: ",
                100 * trainingAcc,
                " % - Validation loss: ",
                validationLoss,
                ", accuracy: ",
                100 * validationAcc,
                " % - Test loss: ",
                testLoss,
                ", accuracy: ",
                100 * testAcc,
                " %",
            )
        end
        return (trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc)
    end

    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (
        trainingLoss,
        trainingAccuracy,
        validationLoss,
        validationAccuracy,
        testLoss,
        testAccuracy,
    ) = calculateMetrics()
    #  y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss)
    push!(trainingAccuracies, trainingAccuracy)
    push!(validationLosses, validationLoss)
    push!(validationAccuracies, validationAccuracy)
    push!(testLosses, testLoss)
    push!(testAccuracies, testAccuracy)

    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0
    bestValidationLoss = validationLoss
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann)

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch < maxEpochs) &&
              (trainingLoss > minLoss) &&
              (numEpochsValidation < maxEpochsVal)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(
            loss,
            params(ann),
            [(trainingInputs', trainingTargets')],
            ADAM(learningRate),
        )
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1
        # Calculamos las metricas en este ciclo
        (
            trainingLoss,
            trainingAccuracy,
            validationLoss,
            validationAccuracy,
            testLoss,
            testAccuracy,
        ) = calculateMetrics()
        #  y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss)
        push!(trainingAccuracies, trainingAccuracy)
        push!(validationLosses, validationLoss)
        push!(validationAccuracies, validationAccuracy)
        push!(testLosses, testLoss)
        push!(testAccuracies, testAccuracy)
        # Aplicamos la parada temprana
        if (validationLoss < bestValidationLoss)
            bestValidationLoss = validationLoss
            numEpochsValidation = 0
            bestANN = deepcopy(ann)
        else
            numEpochsValidation += 1
        end
    end
    return (
        bestANN,
        trainingLosses,
        validationLosses,
        testLosses,
        trainingAccuracies,
        validationAccuracies,
        testAccuracies,
    )
end;
# -------------------------------------------------------------------------
#load image dataset functions

# Functions that allow the conversion from images to Float64 arrays
imageToGrayArray(image::Array{RGB{Normed{UInt8,8}},2}) =
    convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGB{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64,3}(undef, size(image, 1), size(image, 2), 3)
    matrix[:, :, 1] = convert(Array{Float64,2}, red.(image))
    matrix[:, :, 2] = convert(Array{Float64,2}, green.(image))
    matrix[:, :, 3] = convert(Array{Float64,2}, blue.(image))
    return matrix
end;
imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

# Some functions to display an image stored as Float64 matrix
# Overload the existing display function, either for graysacale or color images
import Base.display
display(image::Array{Float64,2}) = display(Gray.(image));
display(image::Array{Float64,3}) = (
    @assert(size(image, 3) == 3); display(RGB.(
        image[:, :, 1],
        image[:, :, 2],
        image[:, :, 3],
    ))
)

# Function to read all of the images in a folder and return them as 2 Float64 arrays: one with color components (3D array) and the other with grayscale components (2D array)
#=function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName));
            # Check that they are color images
            @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
            # Add the image to the vector of images
            push!(images, image);
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images), imageToGrayArray.(images));
end;=#
function loadFolderImages(folderName::String)
    isImageExtension(fileName::String) =
        any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"])
    images = []
    for fileName in readdir(folderName)
        if isImageExtension(fileName)
            image = load(string(folderName, "/", fileName))
            # Check that they are color images
            #@assert(
             #   isa(image, Array{RGB{Normed{UInt8,8}},2}) ||
              #  isa(image, Array{RGB{Normed{UInt8,8}},2})
            #)
            # Add the image to the vector of images
            push!(images, image)
        end
    end
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return images
end;

function processImage(imagen::Array{RGB{Normed{UInt8,8}},2})
    value = 0
    #imagen = load("base_bp_bb.png"); display(imagen);
    #imagen = load("white.png");
    #imagen = imageToColorArray(imagenA)

    # Vamos a detectar los objetos rojos
    #  Aquellos cuyo valor de rojo es superior en cierta cantidad al valor de verde y azul
    # Definimos en que cantidad queremos que sea mayor
    diferenciaRojoVerde = 0.3
    diferenciaRojoAzul = 0.3
    canalRojo = red.(imagen)
    canalVerde = green.(imagen)
    canalAzul = blue.(imagen)
    matrizBooleana =
        (canalRojo .> (canalVerde .+ diferenciaRojoVerde)) .&
        (canalRojo .> (canalAzul .+ diferenciaRojoAzul))
    # Mostramos esta matriz booleana para ver que objetos ha encontrado

    # Esto se podria haber hecho, de forma similar, con el siguiente codigo, definiendo primero la funcion a aplicar en todos los pixeles:
    #esPixelRojo(pixel::RGBA) = (pixel.r > pixel.g + diferenciaRojoVerde) && (pixel.r > pixel.b + diferenciaRojoAzul);
    esPixelRojo(pixel::RGB) = (pixel.r < 0.3) && (pixel.g < 0.3) && (pixel.b < 0.3)
    # Y despues aplicando esa funcion a toda la imagen haciendo un broadcast:
    matrizBooleana = esPixelRojo.(imagen)
    #println(size(imagen));
    parameter = size(imagen)
    sec1 =  parameter[1] ÷ 4;
    sec11= 2*(parameter[1] ÷ 2);
    sec2 =   (parameter[2] ÷ 2);
    sec22 =  3*(parameter[1] ÷ 4);
    imagen1 = augment(imagen, Crop(1:sec1, 1:parameter[2]));
    imagen2 = augment(imagen, Crop(1:sec1, 1:sec2));
    imagen3 = augment(imagen, Crop(sec22:parameter[1], 1:parameter[2]));
    matrizBooleana1 = esPixelRojo.(imagen1)
    matrizBooleana2 = esPixelRojo.(imagen2)
    matrizBooleana3 = esPixelRojo.(imagen3)
    # Aplicamos esta funcion a la matriz booleana (imagen umbralizada) que construimos antes:
    labelArray = ImageMorphology.label_components(matrizBooleana)
    labelArray1 = ImageMorphology.label_components(matrizBooleana1)
    labelArray2 = ImageMorphology.label_components(matrizBooleana2)
    labelArray3 = ImageMorphology.label_components(matrizBooleana3)
    # Calculamos los tamaños
    tamanos = component_lengths(labelArray)
    tamanos1 = component_lengths(labelArray1)
    tamanos2 = component_lengths(labelArray2)
    tamanos3 = component_lengths(labelArray3)
    # Que etiquetas son de objetos demasiado pequeños (30 pixeles o menos):
    etiquetasEliminar = findall(tamanos .<= 10*(parameter[1])) .- 1 # Importate el -1, porque la primera etiqueta es la 0
    etiquetasEliminar1 = findall(tamanos1 .<= 3*(parameter[1]/2)) .- 1 # Importate el -1, porque la primera etiqueta es la 0
    etiquetasEliminar2 = findall(tamanos2 .<= 3*(parameter[1]/2)) .- 1 # Importate el -1, porque la primera etiqueta es la 0
    etiquetasEliminar3 = findall(tamanos3 .<= 3*(parameter[1]/2)) .- 1 # Importate el -1, porque la primera etiqueta es la 0
    # Se construye otra vez la matriz booleana, a partir de la matriz de etiquetas, pero eliminando las etiquetas indicadas
    # Para hacer esto, se hace un bucle sencillo en el que se itera por cada etiqueta
    #  Esto se realiza de forma sencilla con la siguiente linea
    matrizBooleana =
        [!in(etiqueta, etiquetasEliminar) && (etiqueta != 0) for etiqueta in labelArray]

    matrizBooleana1 =
        [!in(etiqueta, etiquetasEliminar1) && (etiqueta != 0) for etiqueta in labelArray1]

    matrizBooleana2 =
        [!in(etiqueta, etiquetasEliminar2) && (etiqueta != 0) for etiqueta in labelArray2]

    matrizBooleana3 =
        [!in(etiqueta, etiquetasEliminar3) && (etiqueta != 0) for etiqueta in labelArray3]

    # Con esos objetos rojos "grandes", se toman de nuevo las etiquetas
    labelArray = ImageMorphology.label_components(matrizBooleana)
    labelArray1 = ImageMorphology.label_components(matrizBooleana1)
    labelArray2 = ImageMorphology.label_components(matrizBooleana2)
    labelArray3 = ImageMorphology.label_components(matrizBooleana3)
    # Cuantos objetos se han detectado:

    # Vamos a situar el centroide de estos objetos en la imagen umbralizada, poniéndolo en color rojo
    # Por tanto, hay que construir una imagen en color:
    imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana)
    # Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
    centroides = ImageMorphology.component_centroids(labelArray)[2:end]
    # Para cada centroide, ponemos su situacion en color rojo
    for centroide in centroides
        x = Int(round(centroide[1]))
        y = Int(round(centroide[2]))
        imagenObjetos[x, y] = RGB(1, 0, 0)
    end

    imagenObjetos1 = RGB.(matrizBooleana1, matrizBooleana1, matrizBooleana1)
    # Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
    centroides1 = ImageMorphology.component_centroids(labelArray1)[2:end]
    # Para cada centroide, ponemos su situacion en color rojo
    for centroide in centroides1
        x = Int(round(centroide[1]))
        y = Int(round(centroide[2]))
        imagenObjetos1[x, y] = RGB(1, 0, 0)
    end

    imagenObjetos2 = RGB.(matrizBooleana2, matrizBooleana2, matrizBooleana2)
    # Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
    centroides2 = ImageMorphology.component_centroids(labelArray2)[2:end]
    # Para cada centroide, ponemos su situacion en color rojo
    for centroide in centroides2
        x = Int(round(centroide[1]))
        y = Int(round(centroide[2]))
        imagenObjetos2[x, y] = RGB(1, 0, 0)
    end

    imagenObjetos3 = RGB.(matrizBooleana3, matrizBooleana3, matrizBooleana3)
    # Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
    centroides3 = ImageMorphology.component_centroids(labelArray3)[2:end]
    # Para cada centroide, ponemos su situacion en color rojo
    for centroide in centroides3
        x = Int(round(centroide[1]))
        y = Int(round(centroide[2]))
        imagenObjetos3[x, y] = RGB(1, 0, 0)
    end

    # Vamos a recuadrar el bounding box de estos objetos, en color verde
    # Calculamos los bounding boxes, y eliminamos el primero (el objeto "0")
    tmp=0;
    value = 0;
    value1 = 0;
    value2 = 0;
    value3 = 0;
    taman3 = 0;
    loc3x=0;
    loc3y=0;

    alto1 = 0;
    boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end]
    for boundingBox in boundingBoxes
        x1 = boundingBox[1][1]
        y1 = boundingBox[1][2]
        x2 = boundingBox[2][1]
        y2 = boundingBox[2][2]
        value =  (x2 - x1) / (y2 - y1);
        imagenObjetos[ x1:x2 , y1 ] .= RGB(0,1,0);
        imagenObjetos[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenObjetos[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenObjetos[ x2 , y1:y2 ] .= RGB(0,1,0);
    end;
    boundingBoxes1 = ImageMorphology.component_boxes(labelArray1)[2:end]
    for boundingBox in boundingBoxes1
        x1 = boundingBox[1][1]
        y1 = boundingBox[1][2]
        x2 = boundingBox[2][1]
        y2 = boundingBox[2][2]
            value1 =  (x2 - x1) / (y2 - y1);
            alto1= (x2 - x1)/parameter[1]
        imagenObjetos1[ x1:x2 , y1 ] .= RGB(0,1,0);
        imagenObjetos1[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenObjetos1[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenObjetos1[ x2 , y1:y2 ] .= RGB(0,1,0);
    end;
    display(imagenObjetos1)
    boundingBoxes2 = ImageMorphology.component_boxes(labelArray2)[2:end]
    for boundingBox in boundingBoxes2
        x1 = boundingBox[1][1]
        y1 = boundingBox[1][2]
        x2 = boundingBox[2][1]
        y2 = boundingBox[2][2]
        value2 =  (x2 - x1) / (y2 - y1);
        imagenObjetos2[ x1:x2 , y1 ] .= RGB(0,1,0);
        imagenObjetos2[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenObjetos2[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenObjetos2[ x2 , y1:y2 ] .= RGB(0,1,0);
    end;
    display(imagenObjetos2)
    boundingBoxes3 = ImageMorphology.component_boxes(labelArray3)[2:end]
    for boundingBox in boundingBoxes3
        x1 = boundingBox[1][1]
        y1 = boundingBox[1][2]
        x2 = boundingBox[2][1]
        y2 = boundingBox[2][2]
            value3 = (x2 - x1) / (y2 - y1);
            taman3 = ((x2 - x1) * (y2 - y1)) / (parameter[1] * parameter[2])
            loc3x = x1/parameter[1];
            loc3y = y1/parameter[2];
            imagenObjetos3[ x1:x2 , y1 ] .= RGB(0,1,0);
            imagenObjetos3[ x1:x2 , y2 ] .= RGB(0,1,0);
            imagenObjetos3[ x1 , y1:y2 ] .= RGB(0,1,0);
            imagenObjetos3[ x2 , y1:y2 ] .= RGB(0,1,0);
    end;
    display(imagenObjetos3)

    return [value value1 value3 alto1 taman3]
end;


function Writem(imagen::Array{RGB{Normed{UInt8,8}},2}, text)
    return text
end;

function loadTrainingDataset()
    barroco = loadFolderImages("Barroco")
    popArt = loadFolderImages("PopArt")

    #targets = [trues(length(positives)); falses(length(negatives))];
    targets = [Writem.((barroco), "Barroco"); Writem.((popArt), "PopArt")]
    return (getInputs(), targets)
end;
#targets = [trues(length(positives)); falses(length(negatives))];
#= targets = [Writem.((pawn), "pawn"); Writem.((queen), "queen"); Writem.((horse), "horse");Writem.((bishop), "bishop"); Writem.((king), "king");Writem.((tower), "tower"); Writem.((charriot), "charriot"); Writem.((elephant), "elephant")]
return ([processImage.(pawn);  processImage.(queen);  processImage.(horse); processImage.(bishop);  processImage.(king); processImage.(tower); processImage.(charriot); processImage.(elephant)], targets)
end; =#


# -------------------------------------------------------------------------
#funciones para la SVM y el kNN

#p4

function confusionMatrix(outputs::Array{Bool,1}, targets::Array{Bool,1})
    #println("outputs");
    #println(outputs);
    #println("targets");
    #println(targets);
    @assert(length(outputs)==length(targets));
    @assert(length(outputs)==length(targets));
    # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
    acc         = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate   = 1 - acc;
    recall      = mean(  outputs[  targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision   = mean(  targets[  outputs]); # Valor predictivo positivo
    NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN, y otros no
    @assert(!isnan(recall) && !isnan(specificity));
    precision   = isnan(precision) ? 0 : precision;
    NPV         = isnan(NPV) ? 0 : NPV;
    # Calculamos F1
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2);
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .&   outputs); # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
    confMatrix[2,1] = sum(  targets .& .!outputs); # FN
    confMatrix[2,2] = sum(  targets .&   outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold), targets);


function confusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    else
        # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
        @assert(all(sum(outputs, dims=2).==1));
        # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
        recall      = zeros(numClasses);
        specificity = zeros(numClasses);
        precision   = zeros(numClasses);
        NPV         = zeros(numClasses);
        F1          = zeros(numClasses);
        # Reservamos memoria para la matriz de confusion
        confMatrix  = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos el numero de patrones de cada clase
        numInstancesFromEachClass = vec(sum(targets, dims=1));
        # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
        #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
        #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
        #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
        for numClass in findall(numInstancesFromEachClass.>0)
            # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
            (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
        end;

        # Reservamos memoria para la matriz de confusion
        confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
        # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
        for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
            # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
            confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
        end;

        # Aplicamos las forma de combinar las metricas macro o weighted
        if weighted
            # Calculamos los valores de ponderacion para hacer el promedio
            weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
            recall      = sum(weights.*recall);
            specificity = sum(weights.*specificity);
            precision   = sum(weights.*precision);
            NPV         = sum(weights.*NPV);
            F1          = sum(weights.*F1);
        else
            # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
            #  En su lugar, realizo la media solamente de las clases que tengan instancias
            numClassesWithInstances = sum(numInstancesFromEachClass.>0);
            recall      = sum(recall)/numClassesWithInstances;
            specificity = sum(specificity)/numClassesWithInstances;
            precision   = sum(precision)/numClassesWithInstances;
            NPV         = sum(NPV)/numClassesWithInstances;
            F1          = sum(F1)/numClassesWithInstances;
        end;
        # Precision y tasa de error las calculamos con las funciones definidas previamente
        acc = accuracy(outputs, targets; dataInRows=true);
        errorRate = 1 - acc;

        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
end;


function confusionMatrix(classes::Array{Any,1}, outputs::Array{Any,1}, targets::Array{Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    #@assert(all([in(output, unique(targets)) for output in outputs]));
    #classes = unique(targets);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted);

# De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);
printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2}; weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs), targets; weighted=weighted);

# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

function oneVSall(inputs::Array{Float64,2}, targets::Array{Bool,2})
    numClasses = size(targets,2);
    # Nos aseguramos de que hay mas de dos clases
    @assert(numClasses>2);
    outputs = Array{Float64,2}(undef, size(inputs,1), numClasses);
    for numClass in 1:numClasses
        model = fit(inputs, targets[:,[numClass]]);
        outputs[:,numClass] .= model(inputs);
    end;
    # Aplicamos la funcion softmax
    outputs = collect(softmax(outputs')');
    # Convertimos a matriz de valores booleanos
    outputs = classifyOutputs(outputs);
    classComparison = (targets .== outputs);
    correctClassifications = all(classComparison, dims=2);
    return mean(correctClassifications);
end;

#p5
function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;
#p6

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Array{Any,1}, numFolds::Int64)

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
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
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];
            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs);

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            #println(testOutputs);
            #println(testTargets);
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(unique(targets), testOutputs, testTargets);

        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:],
                        trainingInputs[validationIndices,:], trainingTargets[validationIndices,:],
                        testInputs,                          testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"],
                        trainingInputs, trainingTargets,
                        testInputs,     testTargets;
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                #println(collect(ann(testInputs')'));
                #println(testTargets);
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;



# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones, con conjuntos de entrenamiento, validacion y test:


    # Fijamos la semilla aleatoria para poder repetir los experimentos
    seed!(11);

    numFolds = 10;

    # Parametros principales de la RNA y del proceso de entrenamiento
    topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
    learningRate = 0.0122; # Tasa de aprendizaje
    numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
    validationRatio = 0.1; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
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

    (dataset, target) = loadTrainingDataset(); #necesito ayuda para convertir los inputs
    inputs =convert(Array{Float64,2}, dataset');
    #inputs = reduce(vcat,dataset);
    targets = convert(Array{Any,1}, target);

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
    #modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, numFolds);

    # Entrenamos los arboles de decision
    #modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, numFolds);

    # Entrenamos los kNN
    #modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, numFolds);
