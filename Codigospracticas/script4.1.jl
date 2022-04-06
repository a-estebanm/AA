function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
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

confusionMatrix(outputs::AbstractArray{<:Real}, targets::Array{Bool,1}; threshold::Float64=0.5) =
    confusionMatrix(Array{Bool,1}(outputs.>=threshold),targets);
