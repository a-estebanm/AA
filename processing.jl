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
    carpetas = readdir("BD")

    n_estilos = length(readdir("BD"))

    hue_mean = Array{Float32,2}(undef, n_estilos , 1)
    sat_mean = Array{Float32,2}(undef, n_estilos, 1)
    val_mean = Array{Float32,2}(undef, n_estilos, 1)
    n_estilo = 0
    for estilo in carpetas
        n_estilo +=1
        fotos = readdir(string("BD/",estilo))
        for foto in fotos
            print(estilo,"/",foto,"\n")
            img = load(string("BD/",estilo,"/",foto))
            img_hsv = HSV.(img)
            channels = channelview(float.(img_hsv))
            hue_img = channels[1,:,:]
            saturation_img = channels[2,:,:]
            value_img = channels[3,:,:]
            hue_mean[n_estilo] = mean(hue_img)
            sat_mean[n_estilo] = mean(saturation_img)
            val_mean[n_estilo] = mean(value_img)
        end
    end

    print("Resultados\n")
    print("media acumulada de hue del barroco:", hue_mean[1],"\n")
    print("media acumulada de saturation del barroco:", sat_mean[1],"\n")
    print("media acumulada de value del barroco:", val_mean[1],"\n")
    print("media acumulada de hue del PopArt:", hue_mean[2],"\n")
    print("media acumulada de saturation del PopArt:", sat_mean[2],"\n")
    print("media acumulada de value del PopArt:", val_mean[2],"\n")
    print("media acumulada de hue del Ukiyo-e:", hue_mean[3],"\n")
    print("media acumulada de saturation del Ukiyo-e:", sat_mean[3],"\n")
    print("media acumulada de value del Ukiyo-e:", val_mean[3],"\n")
    print("media acumulada de hue del Realismo:", hue_mean[4],"\n")
    print("media acumulada de saturation del Realismo:", sat_mean[4],"\n")
    print("media acumulada de value del Realismo:", val_mean[4],"\n")
    print("Final del programa\n")
end

function countPictures(carpetas::Vector{String})
    photoCounter = 0
    for estilo in carpetas
        fotos = readdir(string("BD/",estilo))
        for foto in fotos
            photoCounter += 1
        end
    end
    return photoCounter
end

function getInputs()
    carpetas = readdir("BD")
    photosNum = countPictures(carpetas)
    colors = Array{Any,2}(undef, 6, photosNum)
    aux = 0
    for estilo in carpetas
        fotos = readdir(string("BD/",estilo))
        for foto in fotos
            aux += 1
            img = load(string("BD/",estilo,"/",foto))
            img_hsv = HSV.(img)
            colors[1,aux] = mean(red.(img))
            colors[2,aux] = mean(blue.(img))
            colors[3,aux] = mean(green.(img))
            channels = channelview(float.(img_hsv))
            hue_img = channels[1,:,:]
            saturation_img = channels[2,:,:]
            value_img = channels[3,:,:]
            colors[4,aux] = mean(hue_img)
            colors[5,aux] = mean(saturation_img)
            colors[6,aux] = mean(value_img)
        end
    end
    return colors
end



function getTargets()
    carpetas = readdir("BD")
    photosNum = countPictures(carpetas)
    n_estilos = length(readdir())
    targets = Array{Any,1}(undef,photosNum)
    aux = 0
    for estilo in carpetas
        fotos = readdir(string("BD/",estilo))
        for foto in fotos
            aux += 1
            img = load(string("BD/",estilo,"/",foto))
            targets[aux] = estilo
        end
    end
    return targets
end

function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float32, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float32,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));


function loadImages(folderName::String)
    isImageExtension(fileName::String) = any(uppercase(fileName[end-3:end]) .== [".JPG", ".PNG"]);
    images = [];
    for style in readdir(folderName)
        for fileName in readdir(string(folderName,"/",style))
            if isImageExtension(fileName)
                image = load(string(folderName,"/",style, "/", fileName));
                # Check that they are color images
                @assert(isa(image, Array{RGBA{Normed{UInt8,8}},2}) || isa(image, Array{RGB{Normed{UInt8,8}},2}))
                # Add the image to the vector of images
                push!(images, imresize(image,28,28));
            end;
        end;
    end;
    # Convert the images to arrays by broadcasting the conversion functions, and return the resulting vectors
    return (imageToColorArray.(images));#, imageToGrayArray.(images)); #Lo dejo comentado mientra no sé si lo usaré
end;

#print(matrix)
