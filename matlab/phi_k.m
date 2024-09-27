function phi_k_val = phi_k(k,args)
%PHI_K fourier series coefficients of the spatial distribution
%   Function which returns fourier series coefficients of the spatial 
%   distribution passed as gaussian mixture model in args
% Inputs:
%   k   : current index from the set of indices K
%   args: additional arguments. You must pass at least alpha (mixing 
%         coefficients), D (dimension), Mu (centers of the gaussians), 
%         L (period), Sigma (covariance matrices of the gaussians)
% Outputs: 
%   phi_k_val: length(k)x1 coefficients of the spatial distribution (in
%              Rlength(k))

    

%   if exist(args.Sigma)
%       if or(size(args.Sigma,3)~=length(args.alpha),...
%             length(args.alpha)~=size(args.Mu,2)...
%            )
%           error("myComponent:invalidArgument",strcat("Error. \nThe numbe",... 
%                 "r of matrices Sigma must match the number of mixing coe",...
%                 "fficients alpha and the number of centers Mu (for the g",...
%                 "aussians)"));
%       end
%   end

    Ai={[1 0;0 1],[1 0;0 -1],[-1 0;0 1],[-1 0;0 -1]};
    Am=@(i) cell2mat(Ai(i)); % linear transformation matrices

    phi_k_val=0;

    % iterating gaussians one-by-one in the gaussian mixutre model
    for j=1:length(args.alpha) 

        % phi_k_val are real/even, i.e., iterating up to 2^(args.D-1) is
        % sufficient to characterize the signal (/fully)
        for m=1:2^(args.D-1)
 %          if ~exist(args.Sigma)
 %              switch m
 %                  case 1
 %                      Sigma=args.Sigma1;
 %                  case 2
 %                      Sigma=args.Sigma2;
 %                  case 3
 %                      Sigma=args.Sigma3;
 %                  case 4
 %                      Sigma=args.Sigma4;
 %                  case 5
 %                      Sigma=args.Sigma5;
 %                  case 6
 %                      Sigma=args.Sigma6;
 %                  case 7
 %                      Sigma=args.Sigma7;
 %                  otherwise
 %                      error("Only 7 MX Sigmas currently supported");
 %              end
 %          elseif ~isa(args.Sigma,'casadi.MX')
                Sigma=args.Sigma(:,:,j);
 %          else
 %              Sigma=args.Sigma;
 %          end
            phi_k_val=phi_k_val+(args.alpha(j)/(2^(args.D-1)))*... 
                cos(2*pi*k'*Am(m)*args.Mu(:,j)/args.L).*...
                exp(diag(-2*pi^2*k'*Am(m)*Sigma*...
                    Am(m)'*k/args.L^2)...
                   );
        end
    end

    phi_k_val=phi_k_val/(args.L^args.D);
end

