
function moved = move_poly(poly, move_vec)
    for i  = 1:length(poly)
        poly(i, :)= poly(i,:) + move_vec;
    end
    moved = poly;        
end