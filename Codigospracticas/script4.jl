function confusionMatrix(outputs::Array{Bool,1}, targets::Array{Bool,1})
    @assert(length(outputs)==length(targets));
    # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
    acc = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
    errorRate = 1. - acc;
    recall = mean( outputs[ targets]); # Sensibilidad
    specificity = mean(.!outputs[.!targets]); # Especificidad
    precision = mean( targets[ outputs]); # Valor predictivo positivo
    NPV = mean(.!targets[.!outputs]); # Valor predictivo negativo
    # Controlamos que algunos casos pueden ser NaN
    # Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    # Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    # Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    # Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. 
    # El criterio que se usa aqui es que estos valores seran igual a 0
    # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o los VN son el 100% de los patrones

    # En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
    if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
        recall = 1.;
        precision = 1.;
    elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
        specificity = 1.;
        NPV = 1.;
    end;
    # Ahora controlamos los casos en los que no se han podido evaluar las metricas excluyendo los casos anteriores
    recall = isnan(recall) ? 0. : recall;
    specificity = isnan(specificity) ? 0. : specificity;
    precision = isnan(precision) ? 0. : precision;
    NPV = isnan(NPV) ? 0. : NPV;
    # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
    F1 = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, 2, 2);
    # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    # Primera fila/columna: negativos
    # Segunda fila/columna: positivos
    # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
    confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
    confMatrix[1,2] = sum(.!targets .& outputs); # FP
    # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
    confMatrix[2,1] = sum( targets .& .!outputs); # FN
    confMatrix[2,2] = sum( targets .& outputs); # VP
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix(outputs::Array{Float64,1}, targets::Array{Bool,1};
threshold::Float64=0.5) = confusionMatrix(Array{Bool,1}(outputs.>=threshold),targets);

function confusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2};
   weighted::Bool=true)
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
    recall = zeros(numClasses);
    specificity = zeros(numClasses);
    precision = zeros(numClasses);
    NPV = zeros(numClasses);
    F1 = zeros(numClasses);
    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
    # Calculamos el numero de patrones de cada clase
    numInstancesFromEachClass = vec(sum(targets, dims=1));
    # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
    # Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
    # Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
    # En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), 
    # y no se tendran en cuenta a la hora de unir estas metricas
    for numClass in findall(numInstancesFromEachClass.>0)
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass],targets[:,numClass]);
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
        recall = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision = sum(weights.*precision);
        NPV = sum(weights.*NPV);
        F1 = sum(weights.*F1);
    else
        # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
        # En su lugar, realizo la media solamente de las clases que tengan instancias
        numClassesWithInstances = sum(numInstancesFromEachClass.>0);
        recall = sum(recall)/numClassesWithInstances;

        specificity = sum(specificity)/numClassesWithInstances;
        precision = sum(precision)/numClassesWithInstances;
        NPV = sum(NPV)/numClassesWithInstances;
        F1 = sum(F1)/numClassesWithInstances;
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets; dataInRows=true);
    errorRate = 1 - acc;
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end;
   end;
function confusionMatrix(outputs::Array{Any,1}, targets::Array{Any,1};
    weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique(targets);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes),oneHotEncoding(targets, classes); weighted=weighted);
end;
   confusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2};
   weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets;
   weighted=weighted);
   # De forma similar a la anterior, añado estas funcion porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
   # Con estas funcion se pueden usar indistintamente matrices de Float32 o Float64
   confusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2};
   weighted::Bool=true) = confusionMatrix(convert(Array{Float64,2}, outputs),
   targets; weighted=weighted);
   printConfusionMatrix(outputs::Array{Float32,2}, targets::Array{Bool,2};
   weighted::Bool=true) = printConfusionMatrix(convert(Array{Float64,2}, outputs),
   targets; weighted=weighted);
   # Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::Array{Bool,2}, targets::Array{Bool,2};weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) =
   confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end;
   println(""); );
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
printConfusionMatrix(outputs::Array{Float64,2}, targets::Array{Bool,2};
weighted::Bool=true) = printConfusionMatrix(classifyOutputs(outputs), targets;
weighted=weighted)
# -------------------------------------------------------------------------
# Para probar estas funciones, partimos de los resultados del entrenamiento de la practica anterior
println("Results in the training set:")
trainingOutputs = collect(ann(trainingInputs')');
printConfusionMatrix(trainingOutputs, trainingTargets; weighted=true);
println("Results in the validation set:")
validationOutputs = collect(ann(validationInputs')');
printConfusionMatrix(validationOutputs, validationTargets; weighted=true);
println("Results in the test set:")
testOutputs = collect(ann(testInputs')');
printConfusionMatrix(testOutputs, testTargets; weighted=true);
println("Results in the whole dataset:")
outputs = collect(ann(inputs')');
printConfusionMatrix(outputs, targets; weighted=true);
# -------------------------------------------------------------------------
# Estrategia "uno contra todos" y código de ejemplo:

function oneVSall(inputs::Array{Float64,2}, targets::Array{Bool,2})numClasses = size(targets,2);
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
# A continuacion se muestra de forma practica como se podria usar este esquema de one vs all entrenando RRNNAA en este problema muticlase (flores iris)
# IMPORTANTE: con RR.NN.AA. no es necesario utilizar una estrategia "one vs all" porque ya realiza multiclase
# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion
testRatio = 0.2; # Porcentaje de patrones que se usaran para test
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);
numClasses = size(targets,2);
# Nos aseguramos que el numero de clases es mayor que 2, porque en caso contrario no tiene sentido hacer un "one vs all"
@assert(numClasses>2);
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
# Reservamos memoria para las matrices de salidas de entrenamiento, validacion y test
# En lugar de hacer 3 matrices, voy a hacerlo en una sola con todos los datos
outputs = Array{Float64,2}(undef, size(inputs,1), numClasses);
# Y creamos y entrenamos la RNA con los parametros dados para cada una de las clases
for numClass = 1:numClasses
    # A partir de ahora, no vamos a mostrar por pantalla el resultado de cada ciclo del entrenamiento de la RNA (no vamos a poner el showText=true)
    local ann;
    ann, = trainClassANN(topology,
    trainingInputs, trainingTargets[:,[numClass]],
    validationInputs, validationTargets[:,[numClass]],
    testInputs, testTargets[:,[numClass]];
    maxEpochs=numMaxEpochs, learningRate=learningRate,
    maxEpochsVal=maxEpochsVal);
    # Aplicamos la RNA para calcular las salidas para esta clase concreta y las guardamos en la columna correspondiente de la matriz
    outputs[:,numClass] = ann(inputs')';
end;
# A estas 3 matrices de resultados le pasamos la funcion softmax
# Esto es opcional, y nos vale para poder interpretar la salida de cada modelo como la probabilidad de pertenencia de un patron a una clase concreta
outputs = collect(softmax(outputs')');
# Mostramos las matrices de confusion y las metricas
println("Results in the training set:")
printConfusionMatrix(outputs[trainingIndices,:], trainingTargets;
weighted=true);
println("Results in the validation set:")
printConfusionMatrix(outputs[validationIndices,:], validationTargets;
weighted=true);
println("Results in the test set:")
printConfusionMatrix(outputs[testIndices,:], testTargets; weighted=true);
println("Results in the whole dataset:")
printConfusionMatrix(outputs, targets; weighted=true);
   