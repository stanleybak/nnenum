'''
Split files for latex tables
Stanley Bak
'''

def do_acasxu():
    'do acasxu'

    with open('full_acasxu.dat', 'r') as f:
        lines = f.readlines()

    with open('acasxu-all-front.txt', 'w') as f_front:
        f_front.write('blankheader1 blankheader2\n')
        with open('acasxu-all-results.txt', 'w') as f_results:
            f_results.write('blankheader3\n')
            
            with open('acasxu-all-nnenum.txt', 'w') as f_nnenum:
                with open('acasxu-all-empty.txt', 'w') as f_empty:

                    for line in lines:
                        line = line.strip()
                        if len(line) == 0:
                            continue

                        net, prop, res, secs = line.split('\t')
                        net = net.replace('_', '-')

                        f_front.write(f"{prop} {net}\n")

                        if res == 'safe':
                            res = 'unsat'
                        elif res == 'unsafe':
                            res = 'sat'
                        else:
                            res = 'unknown'
                        
                        f_results.write(f"{res.upper()}\n")

                        secs_str = "{:#.2g}".format(float(secs))

                        f_nnenum.write(f"{secs_str}\n")
                        
                        f_empty.write("-\n")

def do_pat():
    'do pat fully connected'

    nets = ("2", "4", "6")
    eps = ("0.02", "0.05")

    with open('pat-all-front.txt', 'w') as f_front:
        f_front.write('blankheader1 blankheader2 blankheader3\n')
        with open('pat-all-results.txt', 'w') as f_results:
            f_results.write('blankheaderres\n')
            
            with open('pat-all-nnenum.txt', 'w') as f_nnenum:
                with open('pat-all-empty.txt', 'w') as f_empty:

                    for net in nets:
                        for ep in eps:
                            filename = f"summary_pat_net{net}_ep{ep}.dat"
                            
                            with open(filename, 'r') as f:
                                lines = f.readlines()

                            for line in lines:
                                line = line.strip()
                                if len(line) == 0:
                                    break

                                _net_eps, image, res, secs = line.split('\t')

                                f_front.write(f"MNIST{net} {ep} {image}\n")

                                if res == 'safe':
                                    res = 'unsat'
                                elif res == 'unsafe':
                                    res = 'sat'
                                else:
                                    res = 'unknown'
                                    secs = None

                                f_results.write(f"{res.upper()}\n")

                                if secs is None:
                                    secs_str = '-'
                                else:
                                    if 1.0 <= float(secs) < 10.0:
                                        secs_str = f"{round(float(secs), 1)}"
                                    elif float(secs) >= 10.0:
                                        secs_str = f"{round(float(secs))}"
                                    else:
                                        secs_str = "{:#.2f}".format(float(secs))

                                f_nnenum.write(f"{secs_str}\n")

                                f_empty.write("-\n")

def do_eth():
    'do eth cnn'

    types = [('mnist', '0.1'), ('mnist', '0.3'), ('cifar10', '2_255'), ('cifar10', '8_255')]

    with open('ggn-all-front.txt', 'w') as f_front:
        f_front.write('blankheader1 blankheader2 blankheader3\n')
        with open('ggn-all-results.txt', 'w') as f_results:
            f_results.write('blankheaderres\n')
            
            with open('ggn-all-nnenum.txt', 'w') as f_nnenum:
                with open('ggn-all-empty.txt', 'w') as f_empty:

                    for type_str, ep_str in types:
                        filename = f"summary_eth_{type_str}_{ep_str}.dat"

                        ep_str = ep_str.replace('_', '/')

                        with open(filename, 'r') as f:
                            lines = f.readlines()

                        for line in lines:
                            line = line.strip()
                            if len(line) == 0:
                                break

                            _net_eps, image, res, secs, _extra = line.split('\t')

                            f_front.write(f"{type_str.upper()} {ep_str} {image}\n")

                            if res == 'safe':
                                res = 'unsat'
                            elif res == 'unsafe':
                                res = 'sat'
                            else:
                                res = 'unknown'
                                secs = None

                            f_results.write(f"{res.upper()}\n")

                            if secs is None:
                                secs_str = '-'
                            else:
                                if 1.0 <= float(secs) < 10.0:
                                    secs_str = f"{round(float(secs), 1)}"
                                elif float(secs) >= 10.0:
                                    secs_str = f"{round(float(secs))}"
                                else:
                                    secs_str = "{:#.2f}".format(float(secs))

                            f_nnenum.write(f"{secs_str}\n")

                            f_empty.write("-\n")

def main():
    'main entry point'

    acasxu = True
    pat = True
    eth_ggn = True

    if acasxu:
        do_acasxu()
        print("done acasxu")

    if pat:
        do_pat()
        print("done pat")

    if eth_ggn:
        do_eth()
        print("done eth/ggn")
        
if __name__ == '__main__':
    main()
