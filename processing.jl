using Images
using TestImages
using Statistics
using JLD2

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;

function displayStats()
    if (!occursin("BD",pwd())) 
        cd("BD")
    end

    n_estilos = length(readdir())

    red_mean = Array{Float32,2}(undef, n_estilos , 1)
    blue_mean = Array{Float32,2}(undef, n_estilos, 1)
    green_mean = Array{Float32,2}(undef, n_estilos, 1)
    n_estilo = 0
    carpetas = readdir()
    for estilo in carpetas
        n_estilo +=1
        fotos = readdir(estilo)
        for foto in fotos
            print(estilo,"/",foto,"\n")
            img = load(string(estilo,"/",foto))
            red_mean[n_estilo] += mean(red.(img))
            blue_mean[n_estilo] += mean(blue.(img))
            green_mean[n_estilo] += mean(green.(img))
            #auxiliar=imageToColorArray(img)
            #print("AUX:", auxiliar,"\n")
        end
    end

    print("Resultados\n")
    print("media acumulada de rojo del barroco:", red_mean[1],"\n")
    print("media acumulada de azul del barroco:", blue_mean[1],"\n")
    print("media acumulada de verde del barroco:", green_mean[1],"\n")
    print("media acumulada de rojo del PopArt:", red_mean[2],"\n")
    print("media acumulada de azul del PopArt:", blue_mean[2],"\n")
    print("media acumulada de verde del PopArt:", green_mean[2],"\n")
    print("Final del programa\n")

end

function countPictures(carpetas::Vector{String})
    photoCounter = 0
    for estilo in carpetas
        fotos = readdir(estilo)
        for foto in fotos
            photoCounter += 1
        end
    end
    return photoCounter
end

function getInputs()
    if (!occursin("BD",pwd())) 
        cd("BD")
    end
    carpetas = readdir()
    photosNum = countPictures(carpetas)
    n_estilos = length(readdir())
    colors = Array{Any,2}(undef, 3, photosNum)
    aux = 0
    for estilo in carpetas
        fotos = readdir(estilo)
        for foto in fotos
            aux += 1
            img = load(string(estilo,"/",foto))
            colors[1,aux] = mean(red.(img))
            colors[2,aux] = mean(blue.(img))
            colors[3,aux] = mean(green.(img))
        end
    end
    return colors
end

#matrix = getInputs()
#print(matrix)


