using Images
using TestImages
using Statistics

function main() 
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
        end
    end
    loadFolderImages("Barroco/")
    print("Resultados\n")
    print("media acumulada de rojo del barroco:", red_mean[1],"\n")
    print("media acumulada de azul del barroco:", blue_mean[1],"\n")
    print("media acumulada de verde del barroco:", green_mean[1],"\n")
    print("media acumulada de rojo del PopArt:", red_mean[2],"\n")
    print("media acumulada de azul del PopArt:", blue_mean[2],"\n")
    print("media acumulada de verde del PopArt:", green_mean[2],"\n")
    print("Final del programa\n")

end

imageToGrayArray(image:: Array{RGB{Normed{UInt8,8}},2}) = convert(Array{Float64,2}, gray.(Gray.(image)));
imageToGrayArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToGrayArray(RGB.(image));
function imageToColorArray(image::Array{RGB{Normed{UInt8,8}},2})
    matrix = Array{Float64, 3}(undef, size(image,1), size(image,2), 3)
    matrix[:,:,1] = convert(Array{Float64,2}, red.(image));
    matrix[:,:,2] = convert(Array{Float64,2}, green.(image));
    matrix[:,:,3] = convert(Array{Float64,2}, blue.(image));
    return matrix;
end;
imageToColorArray(image::Array{RGBA{Normed{UInt8,8}},2}) = imageToColorArray(RGB.(image));

function loadFolderImages(folderName::String)
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
end;

main()


