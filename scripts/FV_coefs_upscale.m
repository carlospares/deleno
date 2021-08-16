N = 9; % Number of points in the stencil
%left = 2; % Offset of leftmost point in the stencil wrt cell i (stencil: i-l ... i+r)
for k = 1:2
    coefs = '[';
    for left = 0:(N-1)
        right = left + N - 1;
        % idx = left:right;
        dx = 0.1; % final result is dx-independent but removing it from the equations is a hassle

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