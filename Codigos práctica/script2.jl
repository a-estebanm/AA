using DelimitedFiles
using Flux
include("script1.jl")

in=transpose(database)

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

print("\n\nEnd Practice Two\n\n")