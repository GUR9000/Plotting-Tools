'''
Author: Gregor Urban

Contains convenience matplotlib-based plotting functions.

Documentation and code-style improvements are work-in-progress (read: I'll probably not work on it unless asked nicely ;) ).

Most functions are quite well tested, but I won't guarantee anything.
'''


import time
import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import ticker as _ticker
import matplotlib.animation as animation
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def plot(img, title="", normalize=False,  show_value_at_cursor=0, aspect = None, show_colored_even_if_gray = 0,
         save_figure_to = None, DPI=300, colorbar=1, convert_NaNs = False, xlabel = '', ylabel = '', origin_upper = True):

    img = np.squeeze(img)
    if img.ndim==4:
        raise ValueError("if this is a stack of images pass 'g.embedMatrices(img, 2, normalize=1)' instead!")
#        img = embedMatrices(img, 2, normalize = 1)
    if img.ndim==3:
        if img.shape[2]>3:
            img=img.transpose((1,2,0))
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    as_gray = False
    if np.ndim(img)==2 and show_colored_even_if_gray==False:
        as_gray=True
    if title!="":
       fig.suptitle(title, fontsize=20)

    if show_value_at_cursor:
        numcols =img.shape[1]
        numrows =img.shape[0]
        def format_coord(x, y):
            col = int(round(x))
            row = int(round(y))
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = img[row,col]
                zm = np.mean(z)
                return '[ x=%1.4f ,   y=%1.4f ,   z=%s (mean: %1.4f)]'%(x, y, str(z), zm)
            else:
                return '[ x=%1.4f ,   y=%1.4f ]'%(x, y)

    img_post = img.copy()
    if convert_NaNs:
            img_post = nan_to_mean(img_post)

    if normalize:
        mi,ma= np.min(img_post)*1.,np.max(img_post)*1.
        img_post = (img_post-mi)/(ma-mi)

    try:
        imgs = ax.imshow(img_post, cmap=pyplot.cm.Greys_r if as_gray else 'jet', interpolation="nearest", origin='upper' if origin_upper else 'lower')
    except:
        print("ERROR @ plot::ax.imgshow() usd with input {} and dtype {}".format(img_post.shape, img_post.dtype))
        raise
    if show_value_at_cursor:
        ax.format_coord = format_coord
    if colorbar:
        fig.colorbar(imgs)
    if aspect is not None:
        __forceAspect(ax, aspect)


    ax.set_xlabel(xlabel, fontsize=13) #fontsize=int(np.round(font_size_scale*13))
    ax.set_ylabel(ylabel, fontsize=13)

    if save_figure_to is not None:
        pyplot.savefig(save_figure_to, bbox_inches='tight', transparent=True, pad_inches=0, dpi=DPI)
    return imgs


#for name, hex in matplotlib.colors.cnames.iteritems():
#    print(name, hex)
__cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
#'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}

__colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] + list(__cnames.keys()) # white:, 'w']


def _adjustFigAspect(fig,aspect=1):
    """
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    """
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)


def plot_graph(Yaxes=[], Xaxis = None, title ="", xlabel ="x", yaxis_label = "y", ylabels = None, legend_location = 'best',
               xlim=None, ylim=None, save_figure_as = None, DPI = 300,
               sortX = False, line_style='-', linewidth=2, markersize=3, markeredgewidth=0.5,
               logscale = False, transpose = False,  marker = 'o',
               custom_x_ticks_major=[], custom_x_ticks_minor=[], custom_y_ticks_major=[],
                custom_y_ticks_minor=[], figure=None, ax=None, aspect_ratio=None, color_offset = 0,
                bbox_inches='tight', print_warning_and_info = False, override_color = None, font_size_scale = 1., show_major_grid=True):
    """
    Yaxes: list of lists/arrays (for multiple plots in the figure)
    
    Xaxis [OPTIONAL]: list/array
    
    title
    
    xlabel: str

    yaxis_label: str
    
    xlim, ylim: tuple of (min,max) values
        
    ylabels: list of strings; list must have same length as Yaxes!

    sortX: will fix zig-zag problems if x-data is not in order (the plot-points will be the same!! Don't worry)

    line_style: may be a list to overlap multiple plots! :)

    color_offset: int that speciefies the offset in the internal color-scheme array

    override_color: a RGB-color string like '#008080' (==teal)


    Returns:

        figure, ax  ( can be used in another call to plot_graph() to add stuff to the same plot)

    """
    if isinstance(Yaxes[0], float) or isinstance(Yaxes[0], int) or ('numpy.int' in str(type(Yaxes[0]))) or ('numpy.float' in str(type(Yaxes[0]))):
        Yaxes=[Yaxes]
        ylabels=[ylabels]

    if Xaxis is None:

        if len(Yaxes[0])!=2:
            Xaxis=np.arange(len(Yaxes[0]))
        else:
            Xaxis = map(lambda x:x[0], Yaxes)
            Yaxes = [map(lambda x:x[1], Yaxes)]

    if len(Xaxis)!=len(Yaxes[0]) or transpose:

        assert len(Xaxis) == len(Yaxes) # fix / transpose
        Yaxes = np.array(Yaxes)
        Yaxes = np.transpose(Yaxes)

    assert len(Xaxis)==len(Yaxes[0]), "Error: mismatch in number of points --> "+str(len(Xaxis))+" VS "+str(len(Yaxes[0]))

    if sortX:
        indices = np.argsort(Xaxis)
        Xaxis = sorted(Xaxis)
        tmp = []
        for Y in Yaxes:
            tmp.append( [Y[i] for i in indices])
        Yaxes = tmp

    if figure is None or ax is None:
        figure = pyplot.figure()
        ax  = figure.add_subplot(111)
    else:
        if print_warning_and_info:
            print('using given figure and axis')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_axisbelow(True)
    ax.set_facecolor((1., 1., 1.))

    figure.suptitle(title, fontsize=int(np.round(font_size_scale*20)))
    ax.set_xlabel(xlabel, fontsize=int(np.round(font_size_scale*13)))
    ax.set_ylabel(yaxis_label, fontsize=int(np.round(font_size_scale*13)))

    if logscale:
        if print_warning_and_info:
            print("log-scale")
        ax.set_xscale('log')#        pyplot.xscale('log')#            start, end = ax.get_xlim()
        ax.xaxis.set_major_formatter(_ticker.FormatStrFormatter("%d"))

    if aspect_ratio is not None:
        _adjustFigAspect(figure, aspect_ratio)

    if len(custom_x_ticks_minor) or len(custom_y_ticks_minor):
        pyplot.minorticks_on()

    if len(custom_x_ticks_major):
        ax.xaxis.set_ticks(custom_x_ticks_major,minor=0)

    if len(custom_x_ticks_minor):
        ax.xaxis.set_ticks(custom_x_ticks_minor,minor=1)

    if len(custom_y_ticks_major):
        ax.yaxis.set_ticks(custom_y_ticks_major,minor=0)

    if len(custom_y_ticks_minor):
        ax.yaxis.set_ticks(custom_y_ticks_minor,minor=1)

    if len(custom_y_ticks_minor) or len(custom_x_ticks_minor):
        pyplot.grid(b=True, which='minor', color='#dddddd', linestyle='--') #lightgray
    
    if show_major_grid:
        pyplot.grid(b=True, which='major', color='#b0b0b0', linestyle='-')#gray

    if ylabels is not None and len(ylabels):
        show_legend = 1
        assert len(ylabels)==len(Yaxes), 'you did not provide enough or too many labels for the graph: '+str(len(ylabels))+' vs '+str(len(Yaxes))
    else:
        show_legend = 0
        ylabels=["NoLabel_"+str(i) for i in range(len(Yaxes))]
    for i,y,lab in zip(range(len(Yaxes)),Yaxes, ylabels):
        assert len(Xaxis)==len(y)#, "Error: mismatch in number of points --> "+str(len(Xaxis))+" VS "+str(len(Yaxes[0]))
        if isinstance(line_style,list):
            for lis in line_style:
                pyplot.plot(Xaxis, y, (override_color if isinstance(override_color,str) else override_color[i]) if not override_color is None else __colors[(color_offset+i)%len(__colors)], linestyle=lis, marker=marker, markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth, label=lab)
        else:
            pyplot.plot(Xaxis, y, (override_color if isinstance(override_color,str) else override_color[i]) if not override_color is None else __colors[(color_offset+i)%len(__colors)], linestyle=line_style, marker=marker, markersize=markersize, markeredgewidth=markeredgewidth, linewidth=linewidth, label=lab)
    if show_legend:
        ax.legend(loc=legend_location, shadow=0, prop={'size':int(np.round(font_size_scale*11))}, numpoints=1)
    if save_figure_as is not None:
        pyplot.savefig(save_figure_as, bbox_inches = bbox_inches, transparent=True, pad_inches=0, dpi=DPI)
    return figure, ax


def __forceAspect(ax, aspect=1.):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def nan_to_mean(data):
    """
    replaces NaNs with the mean of the data
    """
    mean = np.mean(data[np.isnan(data)==0])
    ret=data.copy()
    ret[np.where(np.isnan(data))] = mean
    return ret


__anim_fig_index=0


def plot_animated(imgs, title="",
         normalize=False,
         aspect = None,
         show_colored_even_if_gray = 0,
         update_interval = 100,
         colorbar=False,
         convert_NaNs = False,
         insert_fake_maxval_into_corner=1):
    """
    animates images (imgs must be a list)

    insert_fake_maxval_into_corner:
        Replace bottom left pixel value with the max pixel value of the entire stack so that plotting will have the correct normalization.

    WARNING:
    -----------

    This will only animate if you save the returned value! Call g.plot_show() to start the animation.
    """
    imgs = np.asarray([np.squeeze(img) for img in imgs])
    new_imgs=[]
    for img in imgs:
        if img.ndim==4:
            raise ValueError("if this is a stack of images pass 'g.embedMatrices(img, 2, normalize=1)' instead!")
            #img = embedMatrices(img, 2, normalize = 1)
        if img.ndim==3:
            if img.shape[2]>3:
                img=img.transpose((1,2,0))
        as_gray = False
        if np.ndim(img)==2 and show_colored_even_if_gray==False:
            as_gray=True
        if convert_NaNs:
                img = nan_to_mean(img)#np.nan_to_num(img_post)
        if normalize:
            mi,ma= np.min(img)*1.,np.max(img)*1.
            img = (img-mi)/(ma-mi+1e-8)
        new_imgs.append(img)
    imgs = new_imgs
    if insert_fake_maxval_into_corner:
        imgs[0][0,0] = 1 if normalize else np.max(imgs)
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    if title!="":
       fig.suptitle(title, fontsize=20)
    try:
        imgshw = ax.imshow(imgs[0], cmap=pyplot.cm.Greys_r if as_gray else pyplot.cm.jet, interpolation="nearest", animated=True)
    except:
        print("plot::ax.imgshow() usd with input:",img.shape)
        raise
    if colorbar:
        fig.colorbar(imgshw)
    if aspect is not None:
        __forceAspect(ax, aspect)

    def updatefig(*args):
        global __anim_fig_index
        __anim_fig_index = (__anim_fig_index+1) % len(imgs)
        imgshw.set_array(imgs[__anim_fig_index])
        return imgshw,

    ani = animation.FuncAnimation(fig, updatefig, interval=update_interval, blit=True)
    return ani


def plot_simple(img, title="", gray=True, normalize=True):
    fig = pyplot.figure()
    if gray:
        pyplot.gray()
    if title!="":
       fig.suptitle(title, fontsize=20)
    pyplot.imshow(img if normalize==0 else img*1./(np.max(img)),interpolation="nearest")
    return


def plot_histogram(data, xlabel='x', ylabel='y', title='', bins = None, normalize_height = False,
                   plot_range = None, figure = None, linewidth = 0.8, save = False, save_figure_to = "default.png", DPI = 300):
    """ returns figure (for potential re-use)"""
    data = np.asarray(data)
    if bins is None:
        bins = int(10 + 3*np.sqrt(np.prod(data.shape)))

    hist, bins = np.histogram(data.flatten(), bins=bins)

    if figure is None:
        figure = pyplot.figure()
    ax  = figure.add_subplot(111)
    if title!="":
       figure.suptitle(title, fontsize=20)
    if normalize_height:
        hist = np.array(hist,'float32')/(np.max(hist))
    if plot_range is not None:
        assert len(plot_range)==2
        ax.set_xlim(plot_range)
    else:
        delt = 0.01*(bins[-1] - bins[0])
        low, high = bins[0] - delt, delt + bins[-1]
        ax.set_xlim((low, high))
    ax.plot(bins[:-1]+ 0.5*(bins[1]-bins[0]), hist, linewidth = linewidth)  #    pyplot.bar(center, hist, align='center', width=width)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if save:
        pyplot.savefig(save_figure_to, bbox_inches='tight', transparent=True, pad_inches=0, dpi=DPI)
    return figure


def plot_surface_3d(data):
    '''
    plot 2D array as 3D surface.
    '''
    assert data.ndim == 2
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    surf = ax.plot_surface(X, Y, data, cmap=pyplot.cm.coolwarm, linewidth=0, antialiased=False)
    return fig


def plot_show(block=True):
    pyplot.show(block=block)


def plot_s_close_all():
    pyplot.close("all")


def show_multiple_figures_add(fig, n, i, image, title, isGray=False):
    """
    Adds <i>th image to figure <fig> as subplot out of <n> images; start index is 0 (optional: grayscale).
    Will plot a graph instead if <image> is a list of two lists: [X_values, Y_values].
    returns subplot/ax.
    """
    x = np.ceil(np.sqrt(n))
    y = np.ceil(n/x)
    if(x*y<n):
        if x<y:
            x+=1
        else:
            y+=1
    ax = fig.add_subplot(x, y, i) #ith subplot in grid x,y
    ax.set_title(title)
    if len(image)==2:
        ax.plot(image[0], image[1], marker='o', markersize=5, markeredgewidth=1, linewidth=1.5, linestyle="-")
    else:
        if isGray:
            plot.gray()
        ax.imshow(image,interpolation='nearest')
    return ax


def _newrange(data, oldrange, soft_tol = 0.45, hard_tol = 0.2, margin = 0.1):
    """
    """
    assert hard_tol > margin and soft_tol >= hard_tol
    mi, ma = np.min(data), np.max(data)
    delta = (ma-mi)
    newi, newa = oldrange
    if mi < oldrange[0] or abs(oldrange[0]-mi) > soft_tol*delta:
        newi = (mi - margin*delta)
    elif abs(oldrange[0]-mi) > hard_tol*delta:
        newi = 0.5*newi + 0.5*(mi - margin*delta)

    if ma > oldrange[1] or abs(oldrange[1]-ma) > soft_tol*delta:
        newa = (ma + margin*delta)
    elif abs(oldrange[1]-ma) > hard_tol*delta:
        newa = 0.5*newa + 0.5*(ma - margin*delta)
    return newi, newa


def combine_images_grayscale(mat, border_width=3, normalize=False, output_ratio=16/9.0, fixed_n_horizontal=0, background_value = 0.5):
    """Creates a large image out of many smaller ones (must all have same shapes!).
    
    creates a single matrix out of smaller ones;
    assumed format : mat[index, i_vert, i_horiz]"""
    mat = np.asarray(mat)
    sh = np.shape(mat)
    if len(sh)==1:
        mat = mat.reshape(sh[0],1,1)
        sh  = np.shape(mat)
    if len(sh)==2:
        mat = mat.reshape(sh[0],int(round(np.sqrt(sh[1]))),int(round(np.sqrt(sh[1]))))
        sh  = np.shape(mat)
    assert len(sh)==3, str(sh)
    n = sh[0]
    if fixed_n_horizontal>0:
        nhor = fixed_n_horizontal
    else:
        nhor = int(np.sqrt(n * output_ratio)) # aim: ratio 16:9
    nvert = int(np.ceil(n* 1.0/nhor))  #warning: too big: nvert*nhor >= n

    if np.prod(sh)==sh[0]:
        ret = background_value*np.ones( (nvert*(border_width+sh[1]) * nhor*(border_width+sh[2]),) ,dtype=np.float32)
        ret[:sh[0]] = mat[:,0,0]
        return ret.reshape((nvert*(border_width+sh[1]) , nhor*(border_width+sh[2])))

    ret = background_value*np.ones( (nvert*(border_width+sh[1]) , nhor*(border_width+sh[2])) ,dtype=np.float32)

    if normalize==True:
        maxs = [np.max(mat[i,:,:])+1e-8 for i in range(n)]
        mins = [np.min(mat[i,:,:]) for i in range(n)]
    else:
        maxs = [1]*n
        mins = [0]*n

    for j in range(nvert):
        for i in range(nhor):
            if i+j*nhor >= n:
                return ret
            ret[j*(border_width+sh[1]):j*(border_width+sh[1])+sh[1] , i*(border_width+sh[2]):i*(border_width+sh[2])+sh[2]] = (mat[i+j*nhor,:,:]-mins[i+j*nhor])/(maxs[i+j*nhor]-mins[i+j*nhor])
    return ret


def combine_images(mat, border_width=3, normalize=False, output_ratio=16.0/9.0, fixed_n_horizontal=0, nChannels=3, background_black=True):
    """Creates a large image out of many smaller ones (must all have same shapes!).
    
    assumed format : mat[index, <any> (e.g. 3 if color), i_vert, i_horiz]
    assuming 3 colors iff 2d input is provided"""
    if isinstance(mat, list):
        mat = np.asarray(mat, 'float32')
    sh = np.shape(mat)

    if len(sh)==2:
        try:
            mat = mat.reshape(sh[0],nChannels,int(round(np.sqrt(sh[1]*1./nChannels))),int(round(np.sqrt(sh[1]*1./nChannels))))
        except:
            mat = mat.reshape(sh[0],1,int(round(np.sqrt(sh[1]))),int(round(np.sqrt(sh[1]))))
        sh  = np.shape(mat)
    assert len(sh)==4
    if sh[1]>sh[3] :
        mat = mat.swapaxes(1,3).swapaxes(2,3)
        sh=mat.shape
    n = sh[0]
    if fixed_n_horizontal>0:
        nhor = fixed_n_horizontal
    else:
        nhor = int(np.sqrt(n * output_ratio)) # aim: ratio 16:9
    nvert = int(n* 1.0/nhor +0.99)  #warning: too big: nvert*nhor >= n

    ret = np.ones( (sh[1],nvert*(border_width+sh[2]) , nhor*(border_width+sh[3])), dtype=np.float32) * (0 if background_black else 1)

    if normalize==True:
        maxs = 1e-9+np.max(mat.reshape(mat.shape[0],-1),axis=1)#[np.max(mat[i,...]) for i in range(n)]
        mins = np.min(mat.reshape(mat.shape[0],-1),axis=1)#[np.min(mat[i,...]) for i in range(n)]
    else:
        maxs = [1]*n
        mins = [0]*n

    for j in range(nvert):
        for i in range(nhor):
            if i+j*nhor >= n:
                return ret
            ret[:,j*(border_width+sh[2]):j*(border_width+sh[2])+sh[2] , i*(border_width+sh[3]):i*(border_width+sh[3])+sh[3]] = (mat[i+j*nhor] - mins[i+j*nhor])/ (maxs[i+j*nhor] - mins[i+j*nhor])
    return ret


def convert_to_image(tensor, max_aspect_ratio = 5.):
    '''
    Turns a given n-dim tensor into something that can be plotted as a 2D image.
    Will use reshaping and padding if necessary and try to produce something square-ish.

    Will try to produce an RGB-color image if any dimension has length 3 and it seems appropriate.
    Will try to preserve the longest edge if it seems appropriate (within <max_aspect_ratio>).
    '''
    MAKE_RGB = 0
    tensor = np.squeeze(tensor)
    sp = np.asarray(tensor.shape)
    if tensor.ndim == 2:
        if max(sp)*1./min(sp) < max_aspect_ratio:
            return tensor

    if np.any(sp == 3) and tensor.ndim >= 3:
        if tensor.ndim == 3:
            if max(sp)*1./min(sp) < max_aspect_ratio:
                return tensor
        MAKE_RGB = 1
        RGB_AXIS = np.argmax(sp==3)  # first entry with a 3

    # preserve longest edge if possible.
    if 1./(max_aspect_ratio * 3**MAKE_RGB) <= np.prod(sp)*1./max(sp)**2 <= max_aspect_ratio * 3**MAKE_RGB:
        preserved_axis = np.argmax(sp)

        if MAKE_RGB:
            axes_ordering = [preserved_axis] + [x for x in list(range(tensor.ndim)) if x not in [preserved_axis, RGB_AXIS]] + [RGB_AXIS]
            print(tensor.shape, axes_ordering)
            tensor = np.transpose(tensor, axes_ordering)
            return tensor.reshape(sp[preserved_axis], -1, 3)
        else:
            axes_ordering = [preserved_axis] + [x for x in list(range(tensor.ndim)) if x != preserved_axis]
            tensor = np.transpose(tensor, axes_ordering)
            return tensor.reshape(sp[preserved_axis], -1)

    # full flatten, preserve RGB
    if MAKE_RGB:
        other = [x for x in list(range(tensor.ndim)) if x != RGB_AXIS]
        tensor = np.transpose(tensor, other + [RGB_AXIS])
        tensor = tensor.reshape(-1, 3)
    else:
        MAKE_RGB = 0
        tensor = tensor.reshape(-1)

    side_dim = int(np.round(np.sqrt(tensor.shape[0])+0.5))
    out = np.zeros([side_dim, side_dim] + ([3] if MAKE_RGB else []), tensor.dtype)
    for i in range(side_dim):
        sl_ce = tensor[i*side_dim:(i+1)*side_dim, ...]
        out[i, :len(sl_ce), ...] = sl_ce
    return out


class RealTimePlotting(object):

    def __init__(self, max_update_FPS = 20, title='RTP_figure'):
        """
        Suggested: call once: g.pyplot.ion()
        """
        pyplot.ion()
        self._img = None
        self._last_update_t = -1
        self._update_interval = 1./max_update_FPS
        self._title = title
        self._prev_val_min_max=[0,1e-6]


    def update(self, data, force_draw = False, title = None, autoscale=True):
        """
        function will not update data/redraw image if doing this would otherwise surpass the <max_update_FPS> setting.
        Call <can_update()> to check if the figure will be updated when calling update()

        force_draw:
            ignore <max_update_FPS>
        """
        if force_draw ==0 and self.can_update()==0:
            return 0
        if data is None:
            if title and self._img:
                self._figure.suptitle(title, fontsize=22)
                pyplot.draw()
                pyplot.pause(0.0001)
            return 1
        self._last_update_t = time.time()
        if self._img is None or not pyplot.fignum_exists(self._figure.number):
            self._figure = pyplot.figure()
            self._img = pyplot.imshow(data, interpolation="nearest", cmap="seismic", aspect='auto') #RdBu jet
            self._figure = pyplot.gcf()
            self._figure.suptitle(self._title, fontsize=22)
        else:
            self._img.set_data(data)

            if autoscale:
                mi,ma = data.min(), data.max()+1e-6
                d1 = (ma-mi)
                if ((self._prev_val_min_max[1]-self._prev_val_min_max[0])/d1 > 1.5 or
                    (self._prev_val_min_max[1]-self._prev_val_min_max[0])/d1 < 0.5 or
                    (self._prev_val_min_max[0]-mi)/d1 > 0.5):
                    self._prev_val_min_max = mi,ma
                    self._img.set_clim(mi,ma)

        if title is not None:
            self._figure.suptitle(title, fontsize=22)
        pyplot.draw()
        pyplot.pause(0.0001)
        return 1

    def can_update(self):
        return (time.time() - self._last_update_t) > self._update_interval


class RealTimePlotting_Graph(object):

    def __init__(self, max_update_FPS = 20, title='RTP_Graph', xlabel='x', ylabel='y', show_lines=True, show_markers=True):
        """
        Suggested: call once: g.pyplot.ion()
        """
        pyplot.ion()
        self._figure = None
        self._last_update_t = -1
        self._update_interval = 1./max_update_FPS
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._show_lines   = show_lines
        self._show_markers = show_markers

    def update(self, y, x=None, force_draw = False, title = None, autoscale=True):
        """
        y can be a single list or list of lists (or matrix) to draw multiple graphs on the same axis.

        function will not update data/redraw image if doing this would otherwise surpass the <max_update_FPS> setting.
        Call <can_update()> to check if the figure will be updated when calling update()

        force_draw:
            ignore <max_update_FPS>
        """
        if force_draw==0 and self.can_update()==0:
            return 0
        self._last_update_t = time.time()
        try:
            assert len(y[0]) >= 1
        except:
            y=[y]
        if x is None:
            x = np.arange(len(y[0]))
        for _y in y:
            assert len(_y)==len(x), "Error: mismatch in number of points --> X=" + str(len(x)) + " VS Y=" + str(len(_y))
        if self._figure is None or not pyplot.fignum_exists(self._figure.number):
            self._figure, self._ax = pyplot.subplots(1,1)
            #self._ax.hold(True) # hold removed in new and "improved" matplotlib
            self._ax.set_xlim((np.min(x), np.max(x)))
            self._ax.set_ylim((np.min(y), np.max(y)))
#            self._ax.set_ylim( min([np.min(_y) for _y in y]), max([np.max(_y) for _y in y]) )
            pyplot.show(False)
            pyplot.draw()
            # self._blitted_background = self._figure.canvas.copy_from_bbox(self._ax.bbox)
            self._points = [self._ax.plot(x[:len(_y)], _y, marker='o' if self._show_markers else 'None', markersize=5,
                                            markeredgewidth=1, linewidth=1.5, linestyle="-" if self._show_lines else "None")[0] for _y in y]
            # self._figure = pyplot.gcf()
            self._figure.suptitle(self._title if title is None else title, fontsize=22)
            self._ax.set_xlabel(self._xlabel, fontsize=13)
            self._ax.set_ylabel(self._ylabel, fontsize=13)
        else:
            # self._figure.canvas.restore_region(self._blitted_background)
            # self._points.set_data(x, y)
            for i, _y in enumerate(y):
                self._points[i].set_xdata(x)
                self._points[i].set_ydata(_y)
            # self._ax.draw_artist(self._points)
            # self._figure.canvas.blit(self._ax.bbox)
            if autoscale:
                self._ax.set_xlim(_newrange(x, self._ax.get_xlim()))
                self._ax.set_ylim(_newrange(y, self._ax.get_ylim()))
        pyplot.draw()
        pyplot.pause(0.0001)
        return 1

    def can_update(self):
        return (time.time() - self._last_update_t) > self._update_interval



class RealTimePlotting_Graph_multi(object):
    def __init__(self, max_update_FPS = 20, title='RTP_Graph', xlabel='x', ylabel='y', show_lines=True, show_markers=True):
        """
        Similar to RealTimePlotting_Graph(), but draws all graphs in separate sub-figures instead of a single one. Useful/necessary if the number of points differs a lot.
        """
        pyplot.ion()
        self._figure = None
        self._last_update_t = -1
        self._update_interval = 1./max_update_FPS
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._show_lines   = show_lines
        self._show_markers = show_markers

    def update(self, y, x=None, force_draw = False, title = None, autoscale=True):
        """
        y can be a single list or list of lists (or matrix) to draw multiple graphs on different subplots.

        <x> will be ignored!!! (TODO: fix)

        function will not update data/redraw image if doing this would otherwise surpass the <max_update_FPS> setting.
        Call <can_update()> to check if the figure will be updated when calling update()

        force_draw:
            ignore <max_update_FPS>
        """
        if force_draw==0 and self.can_update()==0:
            return 0
        self._last_update_t = time.time()
        #if np.ndim(y)==1: FAILS if y is not array-ifyable
        try:
            assert len(y[0]) >= 1
        except:
            y=[y]
        if x is None:
            x = np.arange(max([len(yy) for yy in y]))
#        for _y in y:
#            assert len(_y)==len(x), "Error: mismatch in number of points --> X=" + str(len(x)) + " VS Y=" + str(len(_y))
        if self._figure is None or not pyplot.fignum_exists(self._figure.number):
            n_rows = int(np.ceil(len(y)**0.5))
            n_cols = int(np.ceil(len(y)*1./n_rows))
            self._figure, self._axes = pyplot.subplots(n_rows, n_cols)
            self._axes = self._axes.reshape(-1)
            #self._ax.hold(True) # hold removed in new and "improved" matplotlib
            for ax, yy in zip(self._axes, y):
                ax.set_xlim((np.min(x), np.max(x[:len(yy)])))
                ax.set_ylim((np.min(yy), np.max(yy)))
                ax.set_xlabel(self._xlabel, fontsize=13) #maybe move to end?
                ax.set_ylabel(self._ylabel, fontsize=13)
            #pyplot.show(False)
            pyplot.draw()
            # self._blitted_background = self._figure.canvas.copy_from_bbox(self._ax.bbox)

            self._points = [ax.plot(x[:len(yy)], yy, marker='o' if self._show_markers else 'None', markersize=5,
                                    markeredgewidth=1, linewidth=1.5, linestyle="-" if self._show_lines else "None")[0] for ax, yy in zip(self._axes, y)]
            # self._figure = pyplot.gcf()
            self._figure.suptitle(self._title if title is None else title, fontsize=22)
#            self._ax.set_xlabel(self._xlabel, fontsize=13)
#            self._ax.set_ylabel(self._ylabel, fontsize=13)
        else:
            # self._figure.canvas.restore_region(self._blitted_background)
            # self._points.set_data(x, y)
            for i, yy in enumerate(y):
                self._points[i].set_xdata(x[:len(yy)])
                self._points[i].set_ydata(yy)
            # self._ax.draw_artist(self._points)
            # self._figure.canvas.blit(self._ax.bbox)
            if autoscale:
#                self._axes.set_xlim(_newrange(x, self._ax.get_xlim()))
                for ax, yy in zip(self._axes, y):
                    ax.set_ylim(_newrange(yy, ax.get_ylim()))
        pyplot.draw()
        pyplot.pause(0.0001)
        return 1

    def can_update(self):
        return (time.time() - self._last_update_t) > self._update_interval
