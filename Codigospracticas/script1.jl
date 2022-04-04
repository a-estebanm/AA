using DelimitedFiles

dataset = readdlm("AA\\Codigospracticas\\iris.data",','); #Puede que haya que pillar un Path en vez de esto

inputs=dataset[:,1:4];
targets=dataset[:,5];

inputs = convert(Array{Float32,2},inputs);
@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"

clases=unique(targets)
num_class = length(clases)
if (num_class == 2)        # Si solo hay dos clases, se devuelve una matriz con una columna
	cat_targets = Array{Bool,2}(undef, size(targets,1), 1);
	cat_targets[:,1] .= (targets.==clases[1])   
else
	cat_targets = Array{Bool,2}(undef, size(targets,1), num_class)      
	for num = 1:num_class            
		cat_targets[:,num_class] .= (targets.==clases[num])     
	end
end




function normalize_atr(inputs::Array{Float32,1})
	maxv::Float32=maximum(inputs)
	minv::Float32=minimum(inputs)
	return (inputs.-minv)/(maxv-minv)
end

function normalize(inputs::Array{Float32,2})
	finalmatrix=zeros(size(inputs, 1), size(inputs, 2))
	for x in 1:size(inputs, 2)
		finalmatrix[:,x]=normalize_atr(inputs[:,x])
	end
	return finalmatrix
end

norm_inputs=normalize(inputs)

println("End of first script")