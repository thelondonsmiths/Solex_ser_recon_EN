import math
import sys

flag_dictionnary = {
    'h' : 'Help',                   #
    'w' : 'shift',                  # [0]
    'd' : 'flag_display',           # True/False display all pictures
    'x' : 'ratio_fixe',             # None for automatic, 1 for deactivated
    'f' : 'save_fit',               # True/False
    'c' : 'clahe_only',             # True/False
    'p' : 'disk_display',           # True/False protuberances 
    's' : 'crop_width_square',      # True/False
    't' : 'transversalium',         # True/False
    'm' : 'flip_x',                 # True/False
    'r' : 'fixed_width'             # None
}

def usage():
    usage_ = "SHG_MAIN.py [-hwdxfcpstmr] [file(s) to treat, * allowed]\n"
    usage_ += "'h' : 'Help', display help menu.\n"
    usage_ += "'w' : 'a,b,c, ...'  produce images at a, b, c ... pixels.\n"
    usage_ += "'w' : 'x:y:w'  produce images starting at x, finishing at y, every w pixels.\n"    
    usage_ += "'d' : 'flag_display', display all graphics (False by default)\n"
    usage_ += "'x' : 'ratio_fixe', disable ellipse fitting\n"
    usage_ += "'f' : 'save_fit', save all fits files (False by default)\n"
    usage_ += "'c' : 'clahe_only',  only final clahe image is saved (False by default)\n"
    usage_ += "'p' : 'disk_display' turn off black disk with protuberance images (False by default)\n"
    usage_ += "'s' : 'crop_square_width', crop the width to equal the height (False by default)\n"
    usage_ += "'t' : 'disable transversalium', disable transversalium correction (False by default)\n"
    usage_ += "'m' : 'mirror flip', mirror flip in x-direction (False by default)\n"
    usage_ += "'r' : 'w'  crop width to a constant no. of pixels."
    return usage_
    
def treat_flag_at_cli(options, argument):
    """read cli arguments and produce options variable"""
    options['disk_display'] = True # disk_display on by default
    #reading arguments
    i=0
    while i < len(argument[1:]): #there's a '-' at first)
        character = argument[1:][i]
        if character=='h': #asking help menu
            print(usage())
            sys.exit()
        elif character=='w' :
            #find characters for shifting
            shift=''
            stop = False
            try : 
                while not stop : 
                    if argument[1:][i+1].isdigit() or argument[1:][i+1]==':' or argument[1:][i+1]==',' or argument[1:][i+1]=='-': 
                        shift+=argument[1:][i+1]
                        i+=1
                    else : 
                        i+=1
                        stop=True
            except IndexError :
                i+=1 #the reach the end of arguments.
            shift_choice = shift.split(':')
            if len(shift_choice) == 1:
                options['shift'] = list(map(int, [x.strip() for x in shift.split(',')]))
            elif len(shift_choice) == 2:
                options['shift'] = list(range(int(shift_choice[0].strip()), int(shift_choice[1].strip())+1))
            elif len(shift_choice) == 3:
                options['shift'] = list(range(int(shift_choice[0].strip()), int(shift_choice[1].strip())+1, int(shift_choice[2].strip())))
            else:
                print('invalid shift input')
                sys.exit()
        elif character=='t':
            options['transversalium'] = False
            i+=1
        elif character=='p':
            options['disk_display'] = False
            i+=1
        elif character=='x':
            options['ratio_fixe'] = 1 # no ellipse fit correction will be applied
            i+=1
        elif character=='r':
            fw = ''
            try:
                while argument[1:][i+1].isdigit():
                    fw += argument[1:][i+1]
                    i += 1
                i += 1
            except IndexError:
                i+=1 #the reach the end of arguments.    
            options['fixed_width'] = int(fw)
        else : 
            try : #all others
                options[flag_dictionnary[character]]=True if flag_dictionnary.get(character) else False
                i+=1
            except KeyError: 
                print('ERROR !!! At least one argument is not accepted')
                print(usage())
                i+=1
    print('options %s' % (options))

def handle_CLI(options):
    serfiles = []
    for argument in sys.argv[1:]:
        if '-' == argument[0]: #it's flag options
            treat_flag_at_cli(options, argument)
        else : #it's a file or some files
            if argument.split('.')[-1].upper()=='SER' or argument.split('.')[-1].upper()=='AVI': 
                serfiles.append(argument)
            else:
                print(f'WARNING: {argument} was not a valid SER or AVI file name and was ignored. Remember to use "-" if you want to input a flag')
    print('theses files are going to be processed : ', serfiles)
    return serfiles
        
