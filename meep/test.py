def pass_vor(func, vor):
    def wrapper(*args, **kwargs):
        func(vor, *args, **kwargs)
    return wrapper

def my_eps(my_vor, coord):
    print(my_vor)
    return my_vor

func = pass_vor(my_eps, 'my vor')
print(func('coord'))
