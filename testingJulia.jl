#=
testingJulia:
- Julia version: 
- Author: frasercripps
- Date: 2021-05-30
=#
using Revise
using Distributions

function loopTwo(b)
    for i=1:1000
        b[i] -=1
    end
    return b
end
function loopTest(a)
    for j=1:100
        for i=1:1000
            a[i] +=1
        end
    end
    loopTwo(a)
end


arr = rand(Uniform(1,100), 1, 1000)
@time arrnew = loopTest(arr)
println(arrnew[1:10])


# Taken from: https://rosettacode.org/wiki/Mandelbrot_set#Julia