using DelimitedFiles
using Flux
using Flux.Losses
using Statistics
using Random
include("script2.jl")

#Función para dividir bd en dos subconjuntos
#Input: N número de patrones, P % patrones separados para test
#Output: Tupla de 2 vectores con indices para entrenamiento y test
function holdOut(N::Int, P::Float64)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));

    @assert (size(indices[1:numTrainingInstances],1) + size(indices[numTrainingInstances+1:end]),1)=N;      

    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

#Función para dividir bd en tres subconjuntos
#Input: N número de patrones, Pval % patrones separados para validación y Ptest % patrones separados para test
#Output: Tupla de 3 vectores con indices para entrenamiento, validación y test
function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    #(Entrenamiento+validación) y test
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    #Entrenamiento y validación
    (trainingIndices, validationIndices) =
    holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))

    @assert (trainingValidationIndices[trainingIndices].length+
            trainingValidationIndices[validationIndices].length+
            testIndices.length)=N;

    return (trainingValidationIndices[trainingIndices],trainingValidationIndices[validationIndices], testIndices);
end;

# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de test
function trainClassANN(topology::Array{Int64,1}, training::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}; maxEpochs::Int64=1000, minLoss::Float64=0.0,
    learningRate::Float64=0.1, showText::Bool=false)
    trainingInputs=training[1];
    trainingTargets=training[2];
    testInputs=test[1];
    testTargets=test[2];
    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en
    # entrenamiento como en test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test
    @assert(size(trainingInputs,2)==size(testInputs,2));
    @assert(size(trainingTargets,2)==size(testTargets,2));
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    testLosses = Float64[];
    testAccuracies = Float64[];
    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets');
        testLoss = loss(testInputs', testTargets');

        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna).
        # La matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs');
        testOutputs = ann(testInputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        # Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc = accuracy(trainingOutputs, Array{Bool,2}(trainingTargets'); dataInRows=false);
        testAcc = accuracy(testOutputs, Array{Bool,2}(testTargets'); dataInRows=false);
        # Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc = accuracy(Array{Float64,2}(trainingOutputs'), trainingTargets; dataInRows=true);
        testAcc = accuracy(Array{Float64,2}(testOutputs'), testTargets; dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ",
            accuracy: ", 100*trainingAcc, " % - Test loss: ", testLoss, ", accuracy: ",
            100*testAcc, " %");
        end;
        return (trainingLoss, trainingAcc, testLoss, testAcc)
    end;
    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics();
    # y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    push!(testLosses, testLoss);
    push!(trainingAccuracies, trainingAccuracy);
    push!(testAccuracies, testAccuracy);
    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics();
        # y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss);
        push!(trainingAccuracies, trainingAccuracy);
        push!(testLosses, testLoss);
        push!(testAccuracies, testAccuracy);
    end;

    return (ann, trainingLosses, testLosses, trainingAccuracies, testAccuracies);
end;

# Funcion para entrenar RR.NN.AA. con conjuntos de entrenamiento, validacion y test
# Es la funcion anterior, modificada para calcular errores en el conjunto de
# validacion, y parar el entrenamiento si es necesario
function trainClassANN(topology::Array{Int64,1}, training::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
    validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1,
    maxEpochsVal::Int64=6, showText::Bool=false)

    trainingInputs=training[1];
    trainingTargets=training[2];
    testInputs=test[1];
    testTargets=test[2];
    validationInputs=validation[1];
    validationTargets=validation[2];

    # Se supone que tenemos cada patron en cada fila
    # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validation como test
    @assert(size(trainingInputs,1)==size(trainingTargets,1));
    @assert(size(validationInputs,1)==size(validationTargets,1));
    @assert(size(testInputs,1)==size(testTargets,1));
    # Comprobamos que el numero de columnas coincide en los grupos de
    # entrenamiento, validacion y test

    @assert(size(trainingInputs,2)==size(validationInputs,2)==size(testInputs,2));

    @assert(size(trainingTargets,2)==size(validationTargets,2)==size(testTargets,2));
    # Creamos la RNA
    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2));
    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float64[];
    trainingAccuracies = Float64[];
    validationLosses = Float64[];
    validationAccuracies = Float64[];
    testLosses = Float64[];
    testAccuracies = Float64[];
    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Una funcion util para calcular los resultados y mostrarlos por pantalla
    function calculateMetrics()
        # Calculamos el loss en entrenamiento y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        trainingLoss = loss(trainingInputs', trainingTargets');
        validationLoss = loss(validationInputs', validationTargets');
        testLoss = loss(testInputs', testTargets');

        # Calculamos la salida de la RNA en entrenamiento y test. Para ello hay que pasar la matriz de entradas traspuesta (cada patron en una columna).
        # La matriz de salidas tiene un patron en cada columna
        trainingOutputs = ann(trainingInputs');
        validationOutputs = ann(validationInputs');
        testOutputs = ann(testInputs');
        # Para calcular la precision, ponemos 2 opciones aqui equivalentes:
        # Pasar las matrices con los datos en las columnas. La matriz de salidas ya tiene un patron en cada columna
        trainingAcc = accuracy(trainingOutputs, Array{Bool,2}(trainingTargets'); dataInRows=false);
        validationAcc = accuracy(validationOutputs, Array{Bool,2}(validationTargets'); dataInRows=false);
        testAcc = accuracy(testOutputs, Array{Bool,2}(testTargets'); dataInRows=false);
        # Pasar las matrices con los datos en las filas. Hay que trasponer la matriz de salidas de la RNA, puesto que cada dato esta en una fila
        trainingAcc = accuracy(Array{Float64,2}(trainingOutputs'), trainingTargets; dataInRows=true);
        validationAcc = accuracy(Array{Float64,2}(validationOutputs'), validationTargets; dataInRows=true);
        testAcc = accuracy(Array{Float64,2}(testOutputs'), testTargets; dataInRows=true);
        # Mostramos por pantalla el resultado de este ciclo de entrenamiento si nos lo han indicado
        if showText
            println("Epoch ", numEpoch, ": Training loss: ", trainingLoss, ",
            accuracy: ", 100*trainingAcc, " % - Validation loss: ", validationLoss, ",
            accuracy: ", 100*validationAcc, " % - Test loss: ", testLoss, ", accuracy: ",
            100*testAcc, " %");
        end;
        return (trainingLoss, trainingAcc, validationLoss, validationAcc, testLoss, testAcc)
    end;
    # Calculamos las metricas para el ciclo 0 (sin entrenar nada)
    (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
    # y almacenamos los valores de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    push!(trainingAccuracies, trainingAccuracy);
    push!(validationLosses, validationLoss);
    push!(validationAccuracies, validationAccuracy);
    push!(testLosses, testLoss);
    push!(testAccuracies, testAccuracy);
    # Numero de ciclos sin mejorar el error de validacion y el mejor error de validation encontrado hasta el momento
    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    # Cual es la mejor ann que se ha conseguido
    bestANN = deepcopy(ann);
    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics();
        # y almacenamos los valores de loss y precision en este ciclo
        push!(trainingLosses, trainingLoss);
        push!(trainingAccuracies, trainingAccuracy);
        push!(validationLosses, validationLoss);
        push!(validationAccuracies, validationAccuracy);
        push!(testLosses, testLoss);
        push!(testAccuracies, testAccuracy);
        # Aplicamos la parada temprana
        if (validationLoss<bestValidationLoss)
            bestValidationLoss = validationLoss;
            numEpochsValidation = 0;
            bestANN = deepcopy(ann);
        else
            numEpochsValidation += 1;
        end;
    end;
    return (bestANN, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies);
end;

trainClassANN(topology::Array{Int64,1}, training::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}},
    validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    maxEpochs::Int64=1000, minLoss::Float64=0.0, learningRate::Float64=0.1,
    maxEpochsVal::Int64=6, showText::Bool=false) =
    trainClassANN(topology, (training[1], reshape(training[2], 1)), (validation[1], reshape(validation[2], 1)), (test[1], reshape(test[2], 1)), maxEpochs, minLoss, learningRate)


# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones, con conjuntos de entrenamiento y test:
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
testRatio = 0.2; # Porcentaje de patrones que se usaran para test
# Cargamos el dataset
dataset = readdlm("AA\\Codigospracticas\\iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);
# Creamos los indices de entrenamiento y test
(trainingIndices, testIndices) = holdOut(size(inputs,1), testRatio);
# Dividimos los datos
trainingInputs = inputs[trainingIndices,:];
testInputs = inputs[testIndices,:];

trainingTargets = targets[trainingIndices,:];
testTargets = targets[testIndices,:];
# Calculamos los valores de normalizacion solo del conjunto de entrenamiento
normalizationParams = calculateMinMaxNormalizationParameters(trainingInputs);
# Normalizamos las entradas entre maximo y minimo de forma separada para
#entrenamiento y test, con los parametros hallados anteriormente
normalizeMinMax!(trainingInputs, normalizationParams);
normalizeMinMax!(testInputs, normalizationParams);
# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology,(trainingInputs, trainingTargets),(testInputs, testTargets);
maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true);
# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones, con conjuntos de entrenamiento, validacion y test:
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion
testRatio = 0.2; # Porcentaje de patrones que se usaran para test
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
# Cargamos el dataset
dataset = readdlm("AA\\Codigospracticas\\iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para test
normalizeMinMax!(inputs);
# Creamos los indices de entrenamiento, validacion y test
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1),
validationRatio, testRatio);
# Dividimos los datos
trainingInputs = inputs[trainingIndices,:];
validationInputs = inputs[validationIndices,:];
testInputs = inputs[testIndices,:];
trainingTargets = targets[trainingIndices,:];
validationTargets = targets[validationIndices,:];
testTargets = targets[testIndices,:];

# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology,
    (trainingInputs, trainingTargets), (validationInputs, validationTargets), (testInputs, testTargets);
    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=true);

print("\n\nEnd Practice Three\n\n")
