using DelimitedFiles
using Flux
using Flux.Losses
using Statistics

include("script1.jl")

in=transpose(norm_inputs)

ann = Chain();
ann = Chain(ann..., Dense(9, 5, σ) );
ann = Chain(ann..., Dense(5, 3, identity) );
ann = Chain(ann..., softmax );

loss(in, targets) = Losses.crossentropy(ann(in), targets)

learningRate=0.001

#Flux.train!(loss, params(ann), [(in, targets’)], ADAM(learningRate));

function OneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if(size(classes)==2) #First Case
        aux=Array{Bool}
        for x in size(classes, 1)
            aux[x]=(classes[x].==feature[x])
        end
        reshape(aux,1,:)
        return aux

    elseif(size(classes)>2) #Second Case
        aux=Array{Bool,2}
        reshape(aux, size(classes,1), size(classes,2))
        for x in size(classes,2)
            aux[:,x]=(classes[x].==feature[x])
        end
        return aux

    else print("ERROR") #Error
    end
end

oneHotEncoding(feature::Array{Any,1}) = oneHotEncoding(feature::Array{Any,1},
unique(feature));

oneHotEncoding(feature::Array{Bool,1}) = feature; #Usar reshape?

print("\n\nEnd oneHotEncoding\n\n")

function calculateMinMaxNormalizationParameters(input::AbstractArray{<:Real,2})
    mins::Matrix{Any,1}
    maxs::Matrix{Any,1}
    for x in 1:size(input,2)
        maxv::Float32=maximum(input[:,x])
	    minv::Float32=minimum(input[:,x])

        mins[x]=minv
        maxs[x]=maxv
    end
    return (mins, maxs)
end

function calculateZeroMeanNormalizationParameters(input::AbstractArray{<:Real,2})
    means=zeros(1,size(input,2))
    dts=zeros(1,size(input,2))
    for x in 1:size(input,2)
        meanv::Float32=0
        for y in 1:size(input,1)
            meanv=meanv+input[y,x]
        end

        means[x]=(meanv/size(input,1))

        for y in 1:size(input,1)
            dts[x]=sqrt(((input[y,x])-means[x])^2/size(inputs,1))
        end
    end
    return (means, dts)
end

print("\n\nEnd calculateParameters\n\n")

function normalizeMinMax!(normMatrix::AbstractArray{<:Real,2}, param::NTuple{2, AbstractArray{<:Real,2}})

end

#MIRAR LO DE ARRIBA

#Input: outputs (Salidas de un modelo con un patron por fila)
#Output: outputsBoolean (Matriz de valores bool que indica la clasificación)
function classifyOutputs(outputs::Array{Float64,2}; dataInRows::Bool=true, threshold::Float64=0.5)
    numOutputs = size(outputs, dataInRows ? 2 : 1);
    @assert(numOutputs!=2)
    if numOutputs==1
        #Si tiene una columna, broadcast de >= para generar matriz de vectores bool
        return convert(Array{Bool,2}, outputs.>=threshold);
    else
        #Si tiene más de una crear matriz con true en columna de valor mayor
        
        # Encontrar donde esta el valor mayor
        (_,indicesMaxEachInstance) = findmax(outputs, dims= dataInRows ? 2 : 1);
        # Inicializamos la matriz a falso y cambiamos esos valores
        outputsBoolean = Array{Bool,2}(falses(size(outputs)));
        outputsBoolean[indicesMaxEachInstance] .= true;
        # Verificamos que cada patron solo este en una clase
        @assert(all(sum(outputsBoolean, dims=dataInRows ? 2 : 1).==1));
        return outputsBoolean;
    end;
end;

#Cuatro funciones accuracy que calculen la precisión en un problema de clasificación
#Inputs: targets(Matriz salidas deseadas) y outputs(Salidas emitidas por modelo)

#Función 1
accuracy(outputs::Array{Bool,1}, targets::Array{Bool,1}) = mean(outputs.==targets);

#Función 2
function accuracy(outputs::Array{Bool,2}, targets::Array{Bool,2}; dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            #Si sólo tiene una columna llamada a la anterior función
            return accuracy(outputs[:,1], targets[:,1]);
        else
            #Si tiene más de dos, comparar ambas matrices mirando dónde no coinciden
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=2)
            return mean(correctClassifications)
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            classComparison = targets .== outputs
            correctClassifications = all(classComparison, dims=1)
            return mean(correctClassifications)
        end;
    end;
end;

#Función 3
accuracy(outputs::Array{Float64,1}, targets::Array{Bool,1}; threshold::Float64=0.5) = accuracy(Array{Bool,1}(outputs.>=threshold), targets);

#Funcion 4
function accuracy(outputs::Array{Float64,2}, targets::Array{Bool,2}; 
                    dataInRows::Bool=true)
    @assert(all(size(outputs).==size(targets)));
    if (dataInRows)
        # Cada patron esta en cada fila
        if (size(targets,2)==1)
            return accuracy(outputs[:,1], targets[:,1]);
        else
            return accuracy(classifyOutputs(outputs; dataInRows=true), targets);
        end;
    else
        # Cada patron esta en cada columna
        if (size(targets,1)==1)
            return accuracy(outputs[1,:], targets[1,:]);
        else
            return accuracy(classifyOutputs(outputs; dataInRows=false),
                            targets);
        end;
    end;
end;
# Añado estas funciones porque las RR.NN.AA. dan la salida como matrices de valores Float32 en lugar de Float64
# Con estas funciones se pueden usar indistintamente matrices de Float32 o
Float64
accuracy(outputs::Array{Float32,1}, targets::Array{Bool,1};
threshold::Float64=0.5) = accuracy(Float64.(outputs), targets;
threshold=threshold);
accuracy(outputs::Array{Float32,2}, targets::Array{Bool,2};
dataInRows::Bool=true) = accuracy(Float64.(outputs), targets;
dataInRows=dataInRows);

print("\n\nEnd Practice Two\n\n")

# ------------------------------------- Practica 2
# -------------------------------------------------------
# Funciones para crear y entrenar una RNA
function buildClassANN(numInputs::Int64, topology::Array{Int64,1},
numOutputs::Int64)
ann=Chain();
numInputsLayer = numInputs;
for numOutputLayers = topology
ann = Chain(ann..., Dense(numInputsLayer, numOutputLayers, σ));
numInputsLayer = numOutputLayers;
end;
if (numOutputs == 1)
ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
else
ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
ann = Chain(ann..., softmax);
end;
return ann;
end;

function trainClassANN(topology::Array{Int64,1}, inputs::Array{Float64,2},
targets::Array{Bool,2}; maxEpochs::Int64=1000, minLoss::Float64=0.0,
learningRate::Float64=0.1)
# Se supone que tenemos cada patron en cada fila
# Comprobamos que el numero de filas (numero de patrones) coincide
@assert(size(inputs,1)==size(targets,1));
# Creamos la RNA
ann = buildClassANN(size(inputs,2), topology, size(targets,2));
# Definimos la funcion de loss
loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) :
Losses.crossentropy(ann(x),y);
# Creamos los vectores con los valores de loss y de precision en cada ciclo
trainingLosses = Float64[];
trainingAccuracies = Float64[];
# Empezamos en el ciclo 0
numEpoch = 0;
# Una funcion util para calcular los resultados y mostrarlos por pantalla
function calculateMetrics()
# Calculamos el loss. Para ello hay que pasar las matrices traspuestas(cada patron en una columna)
trainingLoss = loss(inputs', targets');
# Calculamos la salida de la RNA. Para ello hay que pasar la matriz de 
#entradas traspuesta (cada patron en una columna). La matriz de salidas tiene un
#patron en cada columna
outputs = ann(inputs');
# Para calcular la precision, ponemos 2 opciones aqui equivalentes:
# Pasar las matrices con los datos en las columnas. La matriz de
#salidas ya tiene un patron en cada columna
acc = accuracy(outputs, Array{Bool,2}(targets'); dataInRows=false);
# Pasar las matrices con los datos en las filas. Hay que trasponer la
#matriz de salidas de la RNA, puesto que cada dato esta en una fila
acc = accuracy(Array{Float64,2}(outputs'), targets; dataInRows=true);
# Mostramos por pantalla el resultado de este ciclo de entrenamiento
println("Epoch ", numEpoch, ": loss: ", trainingLoss, ", accuracy: ",
100*acc, " %");
return (trainingLoss, acc)
end;
# Calculamos las metricas para el ciclo 0 (sin entrenar nada)
(trainingLoss, trainingAccuracy) = calculateMetrics();
# y almacenamos el valor de loss y precision en este ciclo
push!(trainingLosses, trainingLoss);
push!(trainingAccuracies, trainingAccuracy);
# Entrenamos hasta que se cumpla una condicion de parada
while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
# Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas
#(cada patron en una columna)
Flux.train!(loss, params(ann), [(inputs', targets')],
ADAM(learningRate));
# Aumentamos el numero de ciclo en 1
numEpoch += 1;

# Calculamos las metricas en este ciclo
(trainingLoss, trainingAccuracy) = calculateMetrics()
# y almacenamos el valor de loss y precision en este ciclo
push!(trainingLosses, trainingLoss);
push!(trainingAccuracies, trainingAccuracy);
end;
return (ann, trainingLosses, trainingAccuracies);
end;
# -------------------------------------------------------------------------
# Funciones para calcular los parametros de normalizacion y normalizar
# Para calcular los parametros de normalizacion, segun la forma de normalizar
#que se desee:
calculateMinMaxNormalizationParameters(dataset::Array{Float64,2};
dataInRows=true) =
( minimum(dataset, dims=(dataInRows ? 1 : 2)), maximum(dataset,
dims=(dataInRows ? 1 : 2)) );
calculateZeroMeanNormalizationParameters(dataset::Array{Float64,2};
dataInRows=true) =
( mean(dataset, dims=(dataInRows ? 1 : 2)), std(dataset, dims=(dataInRows ?
1 : 2)) );
# 4 versiones de la funcion para normalizar entre 0 y 1:
# - Nos dan los parametros de normalizacion, y se quiere modificar el array de
#entradas (el nombre de la funcion acaba en '!')
# - No nos dan los parametros de normalizacion, y se quiere modificar el array
#de entradas (el nombre de la funcion acaba en '!')
# - Nos dan los parametros de normalizacion, y no se quiere modificar el array
#de entradas (se crea uno nuevo)
# - No nos dan los parametros de normalizacion, y no se quiere modificar el
#array de entradas (se crea uno nuevo)
function normalizeMinMax!(dataset::Array{Float64,2},
normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
dataInRows=true)
min = normalizationParameters[1];
max = normalizationParameters[2];
dataset .-= min;
dataset ./= (max .- min);
# Si hay algun atributo en el que todos los valores son iguales, se pone a 0
if (dataInRows)
dataset[:, vec(min.==max)] .= 0;
else
dataset[vec(min.==max), :] .= 0;
end
end;
normalizeMinMax!(dataset::Array{Float64,2}; dataInRows=true) =
normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset;
dataInRows=dataInRows); dataInRows=dataInRows);
function normalizeMinMax(dataset::Array{Float64,2},
normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
dataInRows=true)

newDataset = copy(dataset);
normalizeMinMax!(newDataset, normalizationParameters;
dataInRows=dataInRows);
return newDataset;
end;
normalizeMinMax(dataset::Array{Float64,2}; dataInRows=true) =
normalizeMinMax(dataset, calculateMinMaxNormalizationParameters(dataset;
dataInRows=dataInRows); dataInRows=dataInRows);
# 4 versiones similares de la funcion para normalizar de media 0:
# - Nos dan los parametros de normalizacion, y se quiere modificar el array de
#entradas (el nombre de la funcion acaba en '!')
# - No nos dan los parametros de normalizacion, y se quiere modificar el array
#de entradas (el nombre de la funcion acaba en '!')
# - Nos dan los parametros de normalizacion, y no se quiere modificar el array
#de entradas (se crea uno nuevo)
# - No nos dan los parametros de normalizacion, y no se quiere modificar el
#array de entradas (se crea uno nuevo)
function normalizeZeroMean!(dataset::Array{Float64,2},
normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
dataInRows=true)
avg = normalizationParameters[1];
stnd = normalizationParameters[2];
dataset .-= avg;
dataset ./= stnd;
# Si hay algun atributo en el que todos los valores son iguales, se pone a 0
if (dataInRows)
dataset[:, vec(stnd.==0)] .= 0;
else
dataset[vec(stnd.==0), :] .= 0;
end
end;
normalizeZeroMean!(dataset::Array{Float64,2}; dataInRows=true) =
normalizeZeroMean!(dataset, calculateZeroMeanNormalizationParameters(dataset;
dataInRows=dataInRows); dataInRows=dataInRows);
function normalizeZeroMean(dataset::Array{Float64,2},
normalizationParameters::Tuple{Array{Float64,2},Array{Float64,2}};
dataInRows=true)
newDataset = copy(dataset);
normalizeZeroMean!(newDataset, normalizationParameters;
dataInRows=dataInRows);
return newDataset;
end;
normalizeZeroMean(dataset::Array{Float64,2}; dataInRows=true) =
normalizeZeroMean(dataset, calculateZeroMeanNormalizationParameters(dataset;
dataInRows=dataInRows); dataInRows=dataInRows);
# -------------------------------------------------------------------------
# Ejemplo de uso de estas funciones:

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
# Cargamos el dataset
dataset = readdlm("iris.data",',');
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:,1:4]);
targets = oneHotEncoding(dataset[:,5]);
# Comprobamos que las funciones de normalizar funcionan correctamente
# Normalizacion entre maximo y minimo
newInputs = normalizeMinMax(inputs);
@assert(all(minimum(newInputs, dims=1) .== 0));
@assert(all(maximum(newInputs, dims=1) .== 1));
# Normalizacion de media 0. en este caso, debido a redondeos, la media y
#desviacion tipica de cada variable no van a dar exactamente 0 y 1
#respectivamente. Por eso las comprobaciones se hacen de esta manera
newInputs = normalizeZeroMean(inputs);
@assert(all(abs.(mean(newInputs, dims=1)) .<= 1e-10));
@assert(all(abs.(std( newInputs, dims=1)).-1 .<= 1e-10));
# Finalmente, normalizamos las entradas entre maximo y minimo:
normalizeMinMax!(inputs);
# Y creamos y entrenamos la RNA con los parametros dados
(ann, trainingLosses, trainingAccuracies) = trainClassANN(topology, inputs,
targets; maxEpochs=numMaxEpochs, learningRate=learningRate);
