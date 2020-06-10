def solution(a):
    p_list, q_list, diff_list = [], [], []
    for p in range(len(a)-1):
        for q in range(p+1,len(a)):
            p_elem, q_elem = a[p], a[q]
            if p_elem != q_elem:
                if p_elem < q_elem:
                    start, end = p_elem, q_elem
                else:
                    start, end = q_elem, p_elem
                values = [value for value in range(start+1, end)]
                flag = True
                for elem in values:
                    if elem in a:
                        flag = False
                if flag:
                    p_list.append(p)
                    q_list.append(q)
                    print("Index", p,q)
                    print("Element", p_elem,q_elem)
    print(p_list)
    print(q_list)
    if len(p_list) == 0:
        return -1
    else:
        for p,q in zip(p_list, q_list):
            diff_list.append(abs(p-q))
        return min(diff_list)
a = [1,4,7,3,3,5]
distance = solution(a)
print(distance)