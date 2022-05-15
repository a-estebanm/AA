
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean

include("Codigo.jl") 
include("processing.jl")

seed!(10)

funcionTransferenciaCapasConvolucionales = relu;

ann1 = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 3),
    softmax
)

#Duplicamos las neuronas de salida en cada capa conv
ann2 = Chain(
    Conv((3, 3), 3=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 64=>64, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(576, 3),
    softmax
)
#Probamos a eliminar el pad
ann3 = Chain(
    Conv((3, 3), 3=>16, pad=0, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=0, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=0, funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(32, 3),
    softmax
)
#Probamos a tener siempre el mismo numero de canales de entrada y salida
ann4 = Chain(
    Conv((3, 3), 3=>3, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 3=>3, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 3=>3, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(27, 3),
    softmax
)
#reducimos a solo 4 capas
ann5 = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(3136, 3),
    softmax
)
#Probamos a eliminar el resto de capas
ann6 = Chain(
    x -> reshape(x, :, size(x, 4)),
    Dense(2352, 3),
    softmax
)

ann7 = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    x -> reshape(x, :, size(x, 4)),
    Dense(25088, 3),
)

ann8 = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 3),
    softmax
)

ann9 = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 3),
    softmax
)

ann10 = Chain(
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 3),
    softmax
)

anns = []
push!(anns, ann1, ann2, ann3, ann4, ann5, ann6, ann7)
entradaCapa = train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch];
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;
function run_deep(ann)

target = getTargets()

inputs = loadImages("BD")

targets = convert(Array{Any,1}, target);


classes = unique(targets);
#targets = oneHotEncoding(targets, classes);

numFolds = 10

crossValidationIndices = crossvalidation(size(inputs,1), numFolds);
fold_img = 5

train_imgs   = inputs[crossValidationIndices.!=fold_img,:];
train_labels = targets[crossValidationIndices.!=fold_img, :];
test_imgs    = inputs[crossValidationIndices.==fold_img,:];
test_labels  = targets[crossValidationIndices.==fold_img,:];
labels = classes; # Las etiquetas

# Tanto train_imgs como test_imgs son arrays de arrays bidimensionales (arrays de imagenes), es decir, son del tipo Array{Array{Float32,2},1}
#  Generalmente en Deep Learning los datos estan en tipo Float32 y no Float64, es decir, tienen menos precision
#  Esto se hace, entre otras cosas, porque las tarjetas gráficas (excepto las más recientes) suelen operar con este tipo de dato
#  Si se usa Float64 en lugar de Float32, el sistema irá mucho más lento porque tiene que hacer conversiones de Float64 a Float32

# Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz en formato HWCN
#  Es decir, Height x Width x Channels x N
#  En el caso de esta base de datos
#   Height = 28
#   Width = 28
#   Channels = 1 -> son imagenes en escala de grises
#     Si fuesen en color, Channels = 3 (rojo, verde, azul)
# Esta conversion se puede hacer con la siguiente funcion:
function convertirArrayImagenesHWCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 3, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28,3)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,1,i] .= imagenes[i][:,:,1];
        nuevoArray[:,:,2,i] .= imagenes[i][:,:,2];
        nuevoArray[:,:,3,i] .= imagenes[i][:,:,3];

    end;
    return nuevoArray;
end;
train_imgs = convertirArrayImagenesHWCN(train_imgs);
test_imgs = convertirArrayImagenesHWCN(test_imgs);

println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))



println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");



# Definimos la red con la funcion Chain, que concatena distintas capas






# Cuando se tienen tantos patrones de entrenamiento (en este caso 60000),
#  generalmente no se entrena pasando todos los patrones y modificando el error
#  En su lugar, el conjunto de entrenamiento se divide en subconjuntos (batches)
#  y se van aplicando uno a uno

# Hacemos los indices para las particiones
# Cuantos patrones va a tener cada particion
batch_size = 500
# Creamos los indices: partimos el vector 1:N en grupos de batch_size
gruposIndicesBatch = Iterators.partition(1:size(train_imgs,4), batch_size);
println("He creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");


# Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
#  Como primer elemento, las imagenes de ese batch
#     train_imgs[:,:,:,indicesBatch]
#  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
#     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
#     onehotbatch(train_labels[indicesBatch], labels)
#  Por tanto, cada batch será un par dado por
#     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
# Sólo resta iterar por cada batch para construir el vector de batches
train_set = [ (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

# Creamos un batch similar, pero con todas las imagenes de test
test_set = (test_imgs, onehotbatch(test_labels, labels));


# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
train_imgs = nothing;
test_imgs = nothing;
GC.gc(); # Pasar el recolector de basura








    # Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
    # Usaremos como entrada varios patrones de un batch
    numBatchCoger = 1; numImagenEnEseBatch = [12, 6];
    # Para coger esos patrones de ese batch:
    #  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
    #  Por tanto:
    #   train_set[numBatchCoger] -> La tupla del batch seleccionado
    #   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
    #   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch

    numCapas = length(Flux.params(ann));
    println("La RNA tiene ", numCapas, " capas:");
    for numCapa in 1:numCapas
        println("   Capa ", numCapa, ": ", ann[numCapa]);
        # Le pasamos la entrada a esta capa
        global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
        capa = ann[numCapa];
        salidaCapa = capa(entradaCapa);
        println("      La salida de esta capa tiene dimension ", size(salidaCapa));
        entradaCapa = salidaCapa;
    end

    # Sin embargo, para aplicar un patron no hace falta hacer todo eso.
    #  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
    ann(train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch]);




    # Definimos la funcion de loss de forma similar a las prácticas de la asignatura
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
    accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
    # Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


    # Mostramos la precision antes de comenzar el entrenamiento:
    #  train_set es un array de batches
    #  accuracy recibe como parametro un batch
    #  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
    #   y devuelve un array con los resultados
    #  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
    #   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
    println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(train_set)), " %");


    # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
    opt = ADAM(0.001);


    println("Comenzando entrenamiento...")


    while (!criterioFin)

        # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
        global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;
        #print("Loss: ", loss, "\n")
        #print("Ann: ", ann, "\n")
        # Se entrena un ciclo
        Flux.train!(loss, Flux.params(ann), train_set, opt);

        numCiclo += 1;

        # Se calcula la precision en el conjunto de entrenamiento:
        precisionEntrenamiento = mean(accuracy.(train_set));
        #println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(test_set);
            #println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end
    print("Loss: ", loss, "\n")
    print("Ann: ", ann, "\n")
    precisionEntrenamiento = mean(accuracy.(train_set))
    precisionTest = accuracy(test_set);
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");
    println("Precision en el conjunto de test: ", 100*precisionTest, " %")
end

function runDeep()
    print("INICIO DEL PROGRAMA\n")
    num = 0;
    for ann in anns
        num +=1
        if(num == 1 || num == 2 || num == 3 || num ==4 || num == 5 || num == 6)
            continue
        end
        print("ANN número ", num, ":\n")
        global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin, entradaCapa;
        entradaCapa = train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch];
        mejorPrecision = -Inf;
        criterioFin = false;
        numCiclo = 0;
        numCicloUltimaMejora = 0;
        mejorModelo = nothing;
        run_deep(ann)
    end
end


#run(ann1)