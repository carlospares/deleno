N = 3; % Number of points in the stencil
%left = 2; % Offset of leftmost point in the stencil wrt cell i (stencil: i-l ... i+r)
for k = 1:2
    coefs = '[';
    for left = 0:(N-1)
        right = left + N - 1;
        % idx = left:right;
        dx = 1; % fairly sure this cancels out -- dx^(p-1) multiplies LHS and RHS of row p

        A = zeros(N);
        for j = 1:N
            for p = 1:N
                rj = j - left - 1;
                A(j,p) = dx.^(p-1) * ( (rj+1).^p - rj.^p )/p;
            end
        end

        B = zeros(1, N);
        for p = 1:N
            rk = (k - 1)/2;
            B(1,p) = (2/p)*dx.^(p-1) * ((rk + 0.5).^p - rk^p);
        end


        C = B*inv(A);
        temp = ['[' sprintf('%0.16g,', C)];
        temp(end) = ']'; 
        if left < N-1
            temp = [temp ',' newline];
        else
            temp = [temp ']'];
        end
        coefs = [coefs temp];
    end
    disp(coefs)
    disp(newline)
end