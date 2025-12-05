import numpy as np
import os
import re
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from astropy.io import fits
from astropy.visualization import simple_norm

from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.ndimage import shift

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel
from astroquery.ipac.ned import Ned
import astropy.units as u
from astropy.nddata import Cutout2D
# from scipy import interpolate

from IPython.display import clear_output
# from pyBBarolo import *
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

from matplotlib.patches import Rectangle, ConnectionPatch
import warnings
from astropy.wcs import WCS, FITSFixedWarning
from astropy.utils.data import get_pkg_data_filename

from skimage.measure import block_reduce
from scipy.ndimage import zoom
from reproject import reproject_interp

import gc

def create_angle_map(image_size, pa_deg, center_x, center_y):
    pa_rad = np.radians(pa_deg)
    
    x = np.arange(image_size[1])
    y = np.arange(image_size[0])
    X, Y = np.meshgrid(x - center_x, y - center_y)
    
    angle_map = np.arctan2(Y, X)
    angle_map += pa_rad
    angle_map = np.mod(angle_map, 2 * np.pi)
    angle_map_deg = np.degrees(angle_map)
    
    return angle_map_deg

def create_radius_map(hdu, angle_map, dist=20.3, plot=False, incl=32, telescop = "VLT_MUSE"):
    y = np.linspace(0, hdu.data.shape[0], hdu.data.shape[0]) - hdu.header["CRPIX2"]
    x = np.linspace(0, hdu.data.shape[1], hdu.data.shape[1]) - hdu.header["CRPIX1"]

    X, Y = np.meshgrid(x, y)

    def my_function(x, y):
        return np.sqrt(x**2 + y**2)

    my_vectorized_function = np.vectorize(my_function)
    distance_map_pix = my_vectorized_function(X, Y)
    print(f" data.shape = {distance_map_pix.shape}")
    if telescop == "VLT_MUSE":
        CD = hdu.header["CD2_2"]
    elif telescop == "ALMA":
        CD = hdu.header["CDELT2"]
    distance_map = dist * 1000 * np.tan(np.radians(distance_map_pix * CD)) * np.sqrt(1 - (np.cos(angle_map/180.*np.pi)*np.radians(incl))**2)

    if plot:
        im = plt.imshow(distance_map, origin="lower", cmap="rainbow")
        plt.scatter(hdu.header["CRPIX1"], hdu.header["CRPIX2"], c="r")
        plt.contour(distance_map)
        plt.colorbar(im)
        plt.axis("off")
        plt.title("Radius Map (Kpc)")

    return distance_map

def _create_residual_plots(observed_vel, model_vel, resid, vmax_obs=200, vmax_resid=50):
    fig = plt.figure(figsize=[15, 5])
    
    plt.subplot(131)
    # Используйте простые LaTeX выражения
    im = plt.imshow(observed_vel, cmap='coolwarm', origin='lower', 
                   vmin=-vmax_obs, vmax=vmax_obs)
    plt.colorbar(im, label='km/s')
    plt.axis("off")
    plt.title(r"$V_{obs}$")  # Простая версия без \mathrm
    
    plt.subplot(132)
    im = plt.imshow(model_vel, cmap='coolwarm', origin='lower',
                   vmin=-vmax_obs, vmax=vmax_obs)
    plt.colorbar(im, label='km/s')
    plt.axis("off")
    plt.title(r"$V_{model}$")  # Простая версия
    
    plt.subplot(133)
    im = plt.imshow(resid, cmap="coolwarm", origin='lower',
                   vmin=-vmax_resid, vmax=vmax_resid)
    cb = plt.colorbar(im, label='km/s')
    plt.axis("off")
    plt.title("Residuals")  # Английский текст
    
    plt.tight_layout()
    return fig

def compute_velocity_residuals(file, telescop, rad, circ_vel, Flux, dist, PA, incl, sigma_file=None, lim=50, smooth=None, interp="cubic", shift_vel=0, plot=False, save=False, output_dir="."):
    
    
    # Загружаем данные и сохраняем КОПИИ для использования вне контекста
    with fits.open(get_pkg_data_filename(file), memmap=False) as hdul:
        hdu_data = hdul[0].data.copy()
        hdu_header = hdul[0].header.copy()

    scale = np.pi * dist * 1000 / 180 / 3600
    # Теперь используем копии данных вместо оригинальных HDU объектов
    print(f"Shape of data: {hdu_data.shape}")

    if smooth is not None:
        print(hdu_data.shape)
        print(smooth)
        hdu_data = block_reduce(hdu_data, block_size=(smooth, smooth), func=np.nanmedian)
        print(hdu_data.shape)
    # Создаем angle_map используя копии данных
    angle_map = create_angle_map(
        hdu_data.shape, 
        PA, 
        hdu_header["CRPIX1"] / smooth if smooth is not None else hdu_header["CRPIX1"],
        hdu_header["CRPIX2"] / smooth if smooth is not None else hdu_header["CRPIX2"],
    )

    # Создаем временный HDU для функций, которые требуют HDU объект
    temp_hdu = fits.PrimaryHDU(data=hdu_data, header=hdu_header)
    distance_map = create_radius_map(temp_hdu, angle_map, dist=dist, incl=incl, telescop=telescop)
    
    R = np.load(rad) * scale
    V = np.load(circ_vel)
    # Удаляем Nan
    mask = ~np.isnan(R) & ~np.isnan(V)
    R_clean = R[mask]
    V_clean = V[mask]

    # Сортируем
    sort_idx = np.argsort(R_clean)
    R_sorted = R_clean[sort_idx]
    V_sorted = V_clean[sort_idx]

    interpolation_methods = {
        "cubic": lambda: CubicSpline(R_sorted, V_sorted),
        "linear": lambda: interp1d(R_sorted, V_sorted, kind='linear', 
                                  bounds_error=False, fill_value="extrapolate")
    }
    
    
    if interp in interpolation_methods:
        interpolator = interpolation_methods[interp]()
    else:
        raise ValueError(f"Unsupported interpolation: {interp}. "
                        f"Available methods: {list(interpolation_methods.keys())}")
    
    radial_vel = interpolator(distance_map)
    model_vel_map = radial_vel * np.cos(np.radians(angle_map))

    if telescop == "VLT_MUSE":
        if Flux is not None:
            with fits.open(get_pkg_data_filename(Flux), memmap=False) as hdul_flux:
                flux_data = hdul_flux[0].data.copy()
                flux_header = hdul_flux[0].header.copy()
            mask_bad = flux_data > lim
            print(f'{mask_bad.shape, hdu_data.shape} mask and data')
            observed_vel_projected = np.where(mask_bad, (hdu_data - shift_vel)/np.sin(np.radians(incl)), np.nan)
        else:
            observed_vel_projected = (hdu_data - shift_vel)/np.sin(np.radians(incl))

    elif telescop == "ALMA":
        observed_vel_projected = (hdu_data - shift_vel)/np.sin(np.radians(incl))

    residual = observed_vel_projected - model_vel_map

    # if sigma_file is not None:
    #     with fits.open(sigma_file) as hdul_sigma:
    #         sigma_data = hdul_sigma[0].data.copy()  # СОХРАНЯЕМ КОПИЮ ДАННЫХ
    #         print(f"shape of sigma {sigma_data.shape}")  # Используем КОПИЮ
    #     # print(f"shape of sigma {hdu_sigma.data.shape}")
    #     resid = np.where(np.abs(residual) > sigma_data, residual, np.nan)
    #     im = plt.imshow(resid, origin="lower", norm=LogNorm())
        # plt.colorbar(im)
    # else:
    resid = residual

    if plot:
        _create_residual_plots(angle_map, distance_map, model_vel_map, vmax_resid=150)
        _create_residual_plots(observed_vel_projected, model_vel_map, resid, vmax_obs=200, vmax_resid=50)

    if save:
        import re
        import os
        
        galaxy_name = re.search(r"(NGC\d+)", file)
        if galaxy_name:
            galaxy_name = galaxy_name.group(1)
        else:
            galaxy_name = "galaxy"
            
        hdu_header['HISTORY'] = 'Residual velocities calculated'
        hdu_header['BUNIT'] = 'km/s'
        
        # Создаем папку если не существует
        os.makedirs(output_dir, exist_ok=True)
        
        # Сохраняем все файлы с использованием safe_fits_write
        files_to_save = [
            (f"{output_dir}/{galaxy_name}_{telescop}_residuals.fits", resid),
            (f"{output_dir}/{galaxy_name}_{telescop}_circ_velocities_map.fits", model_vel_map),
            (f"{output_dir}/{galaxy_name}_{telescop}_angle_map.fits", angle_map),
            (f"{output_dir}/{galaxy_name}_{telescop}_distance_map.fits", distance_map)
        ]
        
        success_count = 0
        for filename, data in files_to_save:
            if safe_fits_write(filename, data, hdu_header):
                success_count += 1
            else:
                print(f" Не удалось сохранить: {filename}")
        
        print(f" Успешно сохранено {success_count}/{len(files_to_save)} файлов в {output_dir}")
    
    return resid, model_vel_map, fits.PrimaryHDU(data=resid, header=hdu_header)


def close_all_fits_files():
    """Принудительно закрывает все открытые FITS файлы"""
    for obj in gc.get_objects():
        if isinstance(obj, fits.HDUList):
            try:
                obj.close()
                # print("Закрыт открытый FITS файл")
            except:
                pass


def  run_velocity_analysis(galaxy_name, telescop, dist, pa, incl, velocity_file, rad_file, circ_vel_file, flux_file, sigma_file=None, lim=50, smooth=None, interp="cubic", shift_vel=0, plot=False, save=False):
    """
    Основная функция для анализа остаточных скоростей галактики
    """
    output_dir = f"data/{galaxy_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting velocity analysis for {galaxy_name}")
    resid, model_vel_map, hdu = compute_velocity_residuals(
        velocity_file, telescop, rad_file, circ_vel_file, flux_file,  # ← здесь тоже telescop
        dist, pa, incl, sigma_file=sigma_file, lim=lim, interp=interp, smooth=smooth,
        shift_vel=shift_vel, plot=plot, save=save, output_dir=output_dir
    )
    
    print(f"Analysis completed.")
    return resid, model_vel_map, hdu

# def plot_image_with_contour(flux_file, distance_map_file, r, bound, figsize=(6, 4), cmap='grey'):
#     """
#     Рисует изображение с контурами поверх
    
#     Parameters:
#     -----------
#     flux_file : str
#         Путь к FITS файлу с данными изображения
#     distance_map_file : str
#         Путь к FITS файлу с картой расстояний для контуров
#     r : float
#         Основное значение расстояния для контура
#     bound : float
#         Полуширина контура (r ± bound)
#     figsize : tuple, default=(6, 4)
#         Размер фигуры
#     cmap : str, default='grey'
#         Цветовая карта для изображения
    
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         Объект фигуры
#     """
#     # Открываем FITS файлы
#     flux_data = fits.open(get_pkg_data_filename(flux_file), memmap=False)[0].data
#     distance_data = fits.open(get_pkg_data_filename(distance_map_file), memmap=False)[0].data
    
#     # Создаем график
#     fig = plt.figure(figsize=figsize)
#     im = plt.imshow(flux_data, cmap=cmap, origin='lower')
#     plt.colorbar(im)
    
#     # Добавляем контуры
#     contour = plt.contour(distance_data, levels=[r-bound, r+bound], colors='red', linewidths=0.5)
#     plt.clabel(contour, inline=True, fontsize=10)
    
#     plt.title(f'V_r with contour level = {r} kpc')
    
#     return fig


def create_multiple_plots(*arrays, titles=None, cmaps='coolwarm', colorbar_labels='km/s', 
                         vmin_max=None, figsize_per_plot=5, norms='linear', 
                         percent=99.0, share_colorbar=False):
    """
    Create multiple plots for 2D arrays with flexible configuration.
    
    Parameters
    ----------
    *arrays : numpy.ndarray
        Variable number of 2D arrays to plot
    titles : list of str, optional
        Titles for each subplot. If None, will use default titles
    cmaps : str or list of str, optional
        Colormap(s) for plots. Can be single value or list per array
    colorbar_labels : str or list of str, optional
        Label(s) for colorbars. Can be single value or list per array
    vmin_max : tuple, list of tuples, or None, optional
        (vmin, vmax) for color scaling. Can be single tuple or list per array.
        If None, uses simple_norm with percent
    figsize_per_plot : float, optional
        Base size for calculating figure dimensions, by default 5
    norms : str or list of str, optional
        Normalization method(s), by default 'linear'
    percent : float, optional
        Percentile for simple_norm, by default 99.0
    share_colorbar : bool, optional
        Whether to use shared colorbar limits, by default False
    """
    
    n_plots = len(arrays)
    
    # Handle default values and convert to lists
    if titles is None:
        titles = [f'Array {i+1}' for i in range(n_plots)]
    elif isinstance(titles, str):
        titles = [titles] * n_plots
    
    if isinstance(cmaps, str):
        cmaps = [cmaps] * n_plots
    
    if isinstance(colorbar_labels, str):
        colorbar_labels = [colorbar_labels] * n_plots
    
    if isinstance(norms, str):
        norms = [norms] * n_plots
    
    # Handle vmin_max
    if vmin_max is None:
        vmin_max = [None] * n_plots
    elif isinstance(vmin_max, (tuple, list)) and len(vmin_max) == 2 and all(isinstance(x, (int, float)) for x in vmin_max):
        vmin_max = [vmin_max] * n_plots
    
    # Calculate figure size
    fig_width = n_plots * figsize_per_plot
    fig_height = figsize_per_plot
    
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_width, fig_height))
    if n_plots == 1:
        axes = [axes]
    
    # Find global min/max if sharing colorbar
    if share_colorbar and vmin_max[0] is None:
        global_min = min(np.nanpercentile(arr, 100-percent) for arr in arrays)
        global_max = max(np.nanpercentile(arr, percent) for arr in arrays)
        vmin_max = [(global_min, global_max)] * n_plots
    
    images = []
    for i, (arr, ax, title, cmap, cb_label, norm_type, vmm) in enumerate(
            zip(arrays, axes, titles, cmaps, colorbar_labels, norms, vmin_max)):
        
        # Create normalization
        if vmm is None:
            norm = simple_norm(arr, norm_type, percent=percent)
            im = ax.imshow(arr, cmap=cmap, origin='lower', norm=norm)
        else:
            vmin, vmax = vmm
            im = ax.imshow(arr, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cb = plt.colorbar(im, ax=ax)
        cb.set_label(cb_label)
        
        ax.axis('off')
        ax.set_title(title)
        images.append(im)
    
    plt.tight_layout()
    # plt.show()
    
    return fig, axes, images


def plot_with_zoom(array, zoom_area, pdf, title=None, cmap=None, norm_type="log", 
                   log_a=50, percent=70, vmax=1, colorbar_label=rf"$km \, s^{-1} pc^{-1}$",
                   figsize=(10, 5), draw_rectangle=True, draw_connections=True,
                   connection_corners_main=None, connection_corners_zoom=None):
    """
    Plot an array with a zoomed region and optional connecting lines.
    
    Parameters
    ----------
    array : numpy.ndarray
        2D array to plot
    zoom_area : tuple
        (xmin, ymin, xmax, ymax) defining the zoom region
    title : str, optional
        Title for the main plot
    cmap : str, optional
        Colormap for the plots
    norm_type : str, optional
        Normalization type for simple_norm, by default "log"
    log_a : int, optional
        Parameter for log normalization, by default 50
    percent : float, optional
        Percentile for normalization, by default 70
    vmax : float, optional
        Maximum value for color scaling, by default 1
    colorbar_label : str, optional
        Label for the colorbar, by default r"$km \, s^{-1} pc^{-1}$"
    figsize : tuple, optional
        Figure size, by default (20, 10)
    draw_rectangle : bool, optional
        Whether to draw rectangle around zoom area, by default True
    draw_connections : bool, optional
        Whether to draw connecting lines, by default True
    connection_corners_main : list, optional
        Custom corners for main image connections
    connection_corners_zoom : list, optional
        Custom corners for zoom image connections
    """
    
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    # Create normalization
    norm = simple_norm(array, norm_type, log_a=log_a, percent=percent)
    
    # Main plot
    im = ax[0].imshow(array, norm=norm, origin="lower", cmap=cmap)
    im.set_clim(vmax=vmax)
    
    if title is None:
        ax[0].set_title(r"$\nabla V_{\varphi}$")
    else:
        ax[0].set_title(title)
    # ax[0].axis("off")
    
    # Zoomed region
    zoomed = array[zoom_area[1]:zoom_area[3], zoom_area[0]:zoom_area[2]]
    im = ax[1].imshow(zoomed, norm=norm, origin="lower", cmap=cmap)
    cbar = plt.colorbar(im, ax=ax[1])
    cbar.set_label(colorbar_label, rotation=270)
    im.set_clim(vmax=vmax)
    # ax[1].axis("off")
    ax[1].set_title("Zoomed Region")
    
    # Draw rectangle around zoom area
    if draw_rectangle:
        rect = Rectangle((zoom_area[0], zoom_area[1]), 
                        zoom_area[2] - zoom_area[0], 
                        zoom_area[3] - zoom_area[1],
                        linewidth=1, edgecolor='black', facecolor='none')
        ax[0].add_patch(rect)
    
    # Draw connecting lines
    if draw_connections:
        # Default corners if not provided
        if connection_corners_main is None:
            connection_corners_main = [
                (zoom_area[2], zoom_area[1]),  # правый нижний
                (zoom_area[2], zoom_area[3]),  # правый верхний
            ]
        
        if connection_corners_zoom is None:
            connection_corners_zoom = [
                (0, 0),                        # левый нижний
                (0, zoomed.shape[0])           # левый верхний
            ]
        
        # Draw connecting lines for all corners
        for (x1, y1), (x2, y2) in zip(connection_corners_main, connection_corners_zoom):
            con = ConnectionPatch(
                xyA=(x1, y1), coordsA=ax[0].transData,
                xyB=(x2, y2), coordsB=ax[1].transData,
                linestyle="--", linewidth=0.9, color="black",
                arrowstyle="-",
                mutation_scale=20
            )
            fig.add_artist(con)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    # plt.tight_layout()
    plt.show()
    
    return fig, ax



def apply_gradient_sign_by_angle(gradient_map, angle_map, phi1, phi2, pa=222):
    """
    Умножает карту градиентов на -1 для пикселей, где угол попадает в диапазон [phi1, phi2]
    
    Parameters:
    -----------
    gradient_map : numpy.ndarray
        Карта градиентов скоростей
    angle_map : numpy.ndarray
        Карта углов в радианах
    phi1 : float
        Начальный угол диапазона (в радианах)
    phi2 : float
        Конечный угол диапазона (в радианах)
    
    Returns:
    --------
    numpy.ndarray
        Модифицированная карта градиентов
    """
    # Создаем копию исходной карты градиентов
    result_map = gradient_map.copy()
    
    # Создаем маску для углов в заданном диапазоне
    angle_mask = (angle_map >= (pa - phi1)%180) & (angle_map <= (pa - phi2)%180)
    
    # Применяем умножение на -1 для пикселей, попадающих в диапазон углов
    result_map[angle_mask] *= -1
    
    return result_map

def plot_three_panels_advanced(x, *y_arrays, angles=None, values=None, xf=None, err=10, legend,
                              titles=None, ylabels=None, colors=None, yscales=None,
                              figsize=(16, 9), xlim=(0, 360)):
    """
    Расширенная версия функции с поддержкой разного количества графиков
    
    Parameters:
    -----------
    x : array-like
        Данные по оси X для всех графиков
    *y_arrays : array-like
        Переменное количество массивов данных по оси Y
    angles : array-like, optional
        Углы для вертикальных областей
    values : array-like, optional
        Значения для фильтрации углов
    err : float, default=10
        Значение для горизонтальных линий
    titles : list, optional
        Заголовки для графиков
    ylabels : list, optional
        Подписи осей Y
    colors : list, optional
        Цвета для графиков
    figsize : tuple, default=(16, 9)
        Размер фигуры
    xlim : tuple, default=(0, 360)
        Пределы по оси X
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Объект фигуры
    axes : tuple of matplotlib.axes.Axes
        Кортеж с осями графиков
    """
    
    n_plots = len(y_arrays)
   
    # Значения по умолчанию
    if titles is None:
        default_titles = ['Flux', r'$V_{\varphi}$', r'$V_r$', 'Additional']
        titles = default_titles[:n_plots]
    elif len(titles) < n_plots:
        # Дополняем заголовки по умолчанию, если передано недостаточно
        default_titles = ['Flux', r'$V_{\varphi}$', r'$V_r$', 'Plot {}']
        titles = list(titles) + [default_titles[i] if i < 3 else f'Plot {i+1}' 
                                for i in range(len(titles), n_plots)]
    
    if ylabels is None:
        default_ylabels = ['flux', 'km/s', 'km/s', 'value']
        ylabels = default_ylabels[:n_plots]
    elif len(ylabels) < n_plots:
        default_ylabels = ['flux', 'km/s', 'km/s', 'value {}']
        ylabels = list(ylabels) + [default_ylabels[i] if i < 3 else f'value {i+1}' 
                                  for i in range(len(ylabels), n_plots)]
    
    if yscales is None:
        default_yscales = ['log', 'linear', 'linear', 'symlog']
        yscales = default_ylabels[:n_plots]
    elif len(ylabels) < n_plots:
        default_yscales = ['log', 'linear', 'linear', 'symlog']
        yscales = list(yscales) + [default_yscales[i] if i < 3 else f'value {i+1}' 
                                  for i in range(len(yscales), n_plots)]
    

    if colors is None:
        default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        colors = default_colors[:n_plots]
    elif len(colors) < n_plots:
        default_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        colors = list(colors) + default_colors[len(colors):n_plots]
    
    # Создаем сетку графиков
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    
    # Если только один график, преобразуем в список для единообразия
    if n_plots == 1:
        axes = [axes]
    
    # Отрисовываем графики
    for i, (ax, y, color, yscale) in enumerate(zip(axes, y_arrays, colors, yscales)):
        if i==0 and xf is not None:
            ax.scatter(xf, y, color=color, linewidth=0.5 + i*0.5)
        else:
            ax.scatter(x, y, color=color, linewidth=0.5 + i*0.5)
        ax.set_yscale(yscale)
        ax.set_title(titles[i], fontsize=14, pad=15)
        ax.set_ylabel(ylabels[i], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xlim[0], xlim[1])
            
        # Добавляем вертикальные области, если переданы angles и values
        if angles is not None and values is not None:
            for a in angles:
                ax.axvspan(a-15, a+15, color='0.75', zorder=0)
        
        # Горизонтальные линии
        ax.hlines([0, err, -err], xmin=0, xmax=360, color="r", linewidth=1)
    
    fig.legend(handles=[plt.Line2D([0], [0], color=colors[0], linewidth=1.5, label=legend)],
           loc='upper right',
           bbox_to_anchor=(1.0, 1.0),  # правая верхняя точка
           bbox_transform=fig.transFigure,  # координаты относительно всей фигуры
           framealpha=0.8,  # полупрозрачность фона
           edgecolor='black')
    # Настройка расположения
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig, axes

# Функция для безопасной нормализации
def safe_norm(data, method='linear'):
    """Безопасная нормализация данных"""
    data_clean = np.where(np.isfinite(data), data, 0)
    
    if np.all(data_clean == 0):
        return data_clean, None
    
    if method == 'log':
        # Для логарифмической нормализации нужны положительные значения
        positive_data = data_clean[data_clean > 0]
        if len(positive_data) == 0:
            return data_clean, None
        
        vmin = np.percentile(positive_data, 5)
        vmax = np.percentile(positive_data, 95)
        vmin = max(vmin, 1e-6)  # Избегаем нуля для логарифма
        
        try:
            return data_clean, LogNorm(vmin=vmin, vmax=vmax)
        except:
            return data_clean, None
    else:
        # Линейная нормазация
        vmin = np.percentile(data_clean, 5)
        vmax = np.percentile(data_clean, 95)
        return data_clean, Normalize(vmin=vmin, vmax=vmax)


def overlay_multiple_fits(reference_file, other_files, pdf, center_ra=None, center_dec=None, zoom=None, 
                          percents=None, alphas=None, cmaps=None, title=None, mod="overlay",
                       size=200, show_individual=True, image_overlay=True, galaxy_name="galaxy_name"):
    """
    Совмещает несколько FITS-файлов с расширенными опциями
    
    Parameters:
    -----------
    reference_file : str
        Референсный файл
    other_files : list
        Список путей к другим FITS-файлам
    center_ra, center_dec : float, optional
        Координаты центра для отображения
    size : int
        Размер области отображения
    show_individual : bool
        Показывать ли отдельные изображения
    image_overlay: bool
        Создавать ли совмещение
    """
    
    # Загружаем и репроектируем все файлы
    with fits.open(reference_file) as hdul:
        hdu_ref = hdul[0]
        wcs_ref = WCS(hdu_ref.header)
        
        all_data = [hdu_ref.data]
        all_filenames = [reference_file]
    
    for file in other_files:
        try:
            with fits.open(file) as hdul:
                hdu_found = None
                for hdu in hdul:
                    if hdu.data is not None and hasattr(hdu.data, 'shape') and len(hdu.data.shape) >= 2:
                        hdu_found = hdu
                        print(f"Найдены данные в HDU[{hdul.index(hdu)}], форма: {hdu.data.shape}")
                        break
                
                if hdu_found is None:
                    raise ValueError(f"Не найдены данные в файле {file}")

                data_reproj, _ = reproject_interp(hdu_found, hdu_ref.header)
                all_data.append(data_reproj)
                all_filenames.append(file)
        except Exception as e:
            print(f"Пропускаем {file}: {e}")
            continue
    
    # Вырезаем область если задан центр
    if center_ra is not None and center_dec is not None:
        center_pix = wcs_ref.wcs_world2pix(center_ra, center_dec, 0)
        x_center, y_center = int(center_pix[0]), int(center_pix[1])

        data_cut = []
        for data in all_data:
            slice_cut = (slice(max(0, y_center-size//2), min(data.shape[0], y_center+size//2)),
                        slice(max(0, x_center-size//2), min(data.shape[1], x_center+size//2)))
            data_cut.append(data[slice_cut])
    else:
        data_cut = all_data
    

    print(f"data cut shape = {np.asarray(data_cut).shape}")
    if zoom is not None:
        data_cut = np.asarray(data_cut)[:, zoom[1]:zoom[3], zoom[0]:zoom[2]]

    # Создаем изображения
    n_files = len(data_cut)
    
    if percents is None:
        percents = [10] * n_files

    if alphas is None:
        alphas = [0.8] * n_files

    if cmaps is None:
        cmaps = ['gray', 'Oranges', 'Greens', 'GnBu', 'Reds', 'Greens', 'YlOrBr', 'BuPu', 'GnBu']
        # cmaps = ['viridis', 'inferno', 'plasma', 'magma']
        # cmaps = ['viridis', 'tab20c', 'tab20b', 'nipy_spectral', 'gist_ncar']
        # cmaps = ['jet', 'rainbow', 'brg', 'gist_rainbow']
    mask_nan = np.isnan(data_cut[0])

    if show_individual:
        fig_individual = plt.figure(figsize=(4 * n_files, 4))
        
        for i, (data, filename, cmap) in enumerate(zip(data_cut, all_filenames, cmaps)):
            ax = fig_individual.add_subplot(1, n_files, i + 1)
            

            # Теперь безопасный вызов
            vmin = np.nanpercentile(data, 5)
            vmax = np.nanpercentile(data, 95)
            
            im = ax.imshow(data, origin='lower', cmap=cmap, 
                          norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax))
            
            short_name = os.path.basename(filename)
            ax.set_title(short_name, fontsize=9)
            ax.set_xlabel('X [pixels]')
            ax.set_ylabel('Y [pixels]')
            
            plt.colorbar(im, ax=ax, shrink=0.8)
        # fig_individual.savefig(f"pic/{galaxy_name}_.pdf", bbox_inches='tight', dpi=300)
        plt.tight_layout()
        plt.show()
    
    if image_overlay:
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        text1 = TextArea(r'$V_{xy}$', textprops=dict(color='gray', fontsize=14))
        text2 = TextArea(' + ', textprops=dict(color='black', fontsize=14))
        text3 = TextArea(r'$H\alpha$', textprops=dict(color='red', fontsize=14))
        # text4 = TextArea(' + ', textprops=dict(color='black', fontsize=14))
        # text5 = TextArea('dust(JWST f770w)', textprops=dict(color='red', fontsize=14))
        from matplotlib.offsetbox import HPacker
        # packer = HPacker(children=[text1, text2, text3, text4, text5],
        packer = HPacker(children=[text1, text2, text3],
                        sep=5,  # расстояние между элементами
                        pad=0,
                        align='center')

        # Создаем аннотацию
        ann_box = AnnotationBbox(packer, (0.5, 1.02), 
                                xycoords='axes fraction',
                                box_alignment=(0.5, 0),
                                frameon=False)

        ax.add_artist(ann_box)
        ax.axis("off")
        for i, (data, filename, cmap, percent, alpha) in enumerate(zip(data_cut, all_filenames, cmaps, percents, alphas)):
            data_clean, norm = safe_norm(data, 'log')
            if i > 0:
                threshold = np.percentile(data_clean, 100 - percent)
                # Создаем маску
                mask = (data >= threshold) & (~np.isnan(data)) & (~mask_nan)
                data_masked = np.where(mask, data_clean, np.nan)
            else:
                # data_masked = np.where(mask_nan, np.nan, data_clean) 
                data_masked = data_clean 
            
            im = ax.imshow(data_masked, origin='lower', cmap=cmap, norm=norm, alpha=alpha)

            if i==0:
                vmin = np.nanpercentile(data_masked, 5)
                vmax = np.nanpercentile(data_masked, 95)

                im = ax.imshow(data_masked, origin='lower', cmap=cmap, 
                            norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax), alpha=alpha, zorder=1)
                im.set_clim(vmin=0.001, vmax=1)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(f'km/s/pc')
                # ax.contour(data_masked, levels=[0.7, 0.8], zorder=10, colors="white")
            
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.tight_layout()
        plt.show()

    if mod == "mask":
        mask_spiral = data_cut[1].astype(bool)
        masked_data = np.where(~mask_spiral, data_cut[0], np.nan)
        fig_mask, ax = plt.subplots(1, 1, figsize=(6, 4))
        vmin = np.nanpercentile(masked_data, 5)
        vmax = np.nanpercentile(masked_data, 95)
        im = ax.imshow(masked_data, origin='lower', cmap="viridis", 
                          norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax))
            
        
        ax.set_title(short_name, fontsize=9)
        ax.set_xlabel('X [pixels]')
        ax.set_ylabel('Y [pixels]')


    return all_data, [wcs_ref] * n_files, all_filenames


def save_to_pdf(filename, mode='append'):
    """Декоратор для сохранения графиков в PDF
    
    Args:
        filename: имя PDF файла
        mode: 'append' - добавить к существующему, 'overwrite' - перезаписать
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import os
    import time
    from PyPDF2 import PdfMerger, PdfReader, PdfWriter
    import tempfile
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Вызываем оригинальную функцию
            result = func(*args, **kwargs)
            
            # Получаем все активные фигуры
            fig_nums = plt.get_fignums()
            if not fig_nums:
                return result
            
            print(f"Сохраняю {len(fig_nums)} графиков в {filename} (режим: {mode})")
            
            if mode == 'append' and os.path.exists(filename):
                try:
                    # Создаем временный файл для новых графиков
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_new:
                        temp_new_filename = tmp_new.name
                    
                    # Сохраняем новые графики во временный файл
                    with PdfPages(temp_new_filename) as pdf:
                        for fig_num in fig_nums:
                            fig = plt.figure(fig_num)
                            pdf.savefig(fig, bbox_inches='tight')
                    
                    # Ждем чтобы убедиться что файл закрыт
                    time.sleep(0.1)
                    
                    # Создаем временный файл для результата
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_result:
                        temp_result_filename = tmp_result.name
                    
                    # Объединяем существующий PDF с новым
                    merger = PdfMerger()
                    
                    try:
                        # Добавляем оригинальный файл
                        merger.append(filename)
                        
                        # Добавляем новые графики
                        merger.append(temp_new_filename)
                        
                        # Сохраняем результат
                        with open(temp_result_filename, 'wb') as output_file:
                            merger.write(output_file)
                        
                        # Заменяем оригинальный файл
                        os.unlink(filename)
                        os.rename(temp_result_filename, filename)
                        
                        print(f" Успешно добавлено {len(fig_nums)} графиков в {filename}")
                        
                    finally:
                        merger.close()
                    
                except Exception as e:
                    print(f" Ошибка при добавлении в PDF: {e}")
                    # Если не удалось добавить, сохраняем как новый файл
                    print("Пытаюсь сохранить как новый файл...")
                    try:
                        with PdfPages(filename) as pdf:
                            for fig_num in fig_nums:
                                fig = plt.figure(fig_num)
                                pdf.savefig(fig, bbox_inches='tight')
                        print(f"Успешно сохранено как новый файл: {filename}")
                    except Exception as e2:
                        print(f"Ошибка при сохранении нового файла: {e2}")
                
                finally:
                    # Удаляем временные файлы
                    for temp_file in [temp_new_filename, temp_result_filename]:
                        if 'temp_file' in locals() and os.path.exists(temp_file):
                            try:
                                os.unlink(temp_file)
                            except:
                                pass
                
            else:
                # Режим перезаписи или файл не существует
                temp_filename = filename + '.tmp'
                
                try:
                    # Сохраняем во временный файл
                    with PdfPages(temp_filename) as pdf:
                        for fig_num in fig_nums:
                            fig = plt.figure(fig_num)
                            pdf.savefig(fig, bbox_inches='tight')
                    
                    # Ждем чтобы убедиться что файл закрыт
                    time.sleep(0.1)
                    
                    # Заменяем оригинальный файл
                    if os.path.exists(filename):
                        os.unlink(filename)
                    os.rename(temp_filename, filename)
                    
                    action = "перезаписан" if mode == 'overwrite' else "создан"
                    print(f" Успешно {action} файл: {filename}")
                    
                except Exception as e:
                    print(f" Ошибка при сохранении PDF: {e}")
                    # Удаляем временный файл при ошибке
                    if os.path.exists(temp_filename):
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
            
            # Фигуры остаются открытыми для просмотра на экране
            return result
        return wrapper
    return decorator

# def save_to_pdf(filename):
#     """Декоратор для сохранения графиков в PDF с отображением на экране"""
#     from matplotlib.backends.backend_pdf import PdfPages
#     import os
#     import time
    
    # def decorator(func):
    #     def wrapper(*args, **kwargs):
    #         # Вызываем оригинальную функцию
    #         result = func(*args, **kwargs)
            
    #         # Получаем все активные фигуры
    #         fig_nums = plt.get_fignums()
    #         if not fig_nums:
    #             return result
            
    #         print(f"Сохраняю {len(fig_nums)} графиков в {filename}")
            
    #         # Создаем уникальное временное имя файла
    #         temp_filename = filename + '.tmp'
            
    #         try:
    #             # Сохраняем во временный файл
    #             with PdfPages(temp_filename) as pdf:
    #                 for fig_num in fig_nums:
    #                     fig = plt.figure(fig_num)
    #                     pdf.savefig(fig, bbox_inches='tight')
                
    #             # Ждем чтобы убедиться что файл закрыт
    #             time.sleep(0.1)
                
    #             # Заменяем оригинальный файл
    #             if os.path.exists(filename):
    #                 os.unlink(filename)
    #             os.rename(temp_filename, filename)
                
    #             print(f"Успешно сохранено в {filename}")
                
    #         except Exception as e:
    #             print(f"Ошибка при сохранении PDF: {e}")
    #             # Удаляем временный файл при ошибке
    #             if os.path.exists(temp_filename):
    #                 try:
    #                     os.unlink(temp_filename)
    #                 except:
    #                     pass
            
    #         # Фигуры остаются открытыми для просмотра на экране
            
    #         return result
    #     return wrapper
    # return decorator



def plot_image_with_contour(image_file, contour_file, contour_levels, bound,
                           image_cmap='grey', contour_colors='red', 
                           contour_linewidths=0.5, contour_fontsize=10,
                           show_colorbar=True, colorbar_label='Intensity',
                           title=None, figsize=(6, 4), 
                           norm_type="log", percent=90, log_a=1000,
                           origin="lower"):
    """
    Визуализирует изображение с контурными линиями поверх.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        2D массив данных для изображения
    contour_data : numpy.ndarray  
        2D массив данных для контурных линий
    contour_levels : float or list of floats
        Уровни для контурных линий
    image_cmap : str, optional
        Цветовая карта для изображения, по умолчанию 'grey'
    contour_colors : str or list of str, optional
        Цвета для контурных линий, по умолчанию 'red'
    contour_linewidths : float, optional
        Толщина контурных линий, по умолчанию 0.5
    contour_fontsize : int, optional
        Размер шрифта для подписей контуров, по умолчанию 10
    show_colorbar : bool, optional
        Показывать ли цветовую шкалу, по умолчанию True
    colorbar_label : str, optional
        Подпись для цветовой шкалы, по умолчанию 'Intensity'
    title : str, optional
        Заголовок графика, по умолчанию None
    figsize : tuple, optional
        Размер фигуры, по умолчанию (6, 4)
    norm_type : str, optional
        Тип нормализации для изображения, по умолчанию "log"
    percent : float, optional
        Процентиль для нормализации, по умолчанию 90
    log_a : float, optional
        Параметр для логарифмической нормализации, по умолчанию 1000
    origin : str, optional
        Начало координат для imshow, по умолчанию "lower"
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Объект фигуры
    ax : matplotlib.axes.Axes
        Объект осей
    """
    
    image_data = fits.open(get_pkg_data_filename(image_file), memmap=False)[0].data
    contour_data = fits.open(get_pkg_data_filename(contour_file), memmap=False)[0].data
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Создаем нормализацию для изображения
    # norm = simple_norm(image_data, norm_type, percent=percent, log_a=log_a)
    
    # Отображаем основное изображение
    norm = simple_norm(image_data, norm_type, percent=percent)
    im = ax.imshow(image_data, cmap=image_cmap, origin=origin, norm=norm)
    
    # Добавляем контурные линии
    contour = ax.contour(contour_data, levels=[contour_levels-bound, contour_levels+bound],
                        colors=contour_colors, linewidths=contour_linewidths)
    
    # Добавляем подписи к контурным линиям
    ax.clabel(contour, inline=True, fontsize=contour_fontsize)
    
    # Добавляем цветовую шкалу если нужно
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
    
    # Устанавливаем заголовок
    if title:
        ax.set_title(title)
    else:
        # Автогенерация заголовка если не задан
        if isinstance(contour_levels, (list, tuple)) and len(contour_levels) == 2:
            ax.set_title(f'Image with contour levels = [{contour_levels[0]}, {contour_levels[1]}]')
        else:
            ax.set_title(f'Image with contour level = {contour_levels}')
    
    return fig, ax
# Альтернативная версия с явным указанием y массивов для обратной совместимости
def plot_multiple_panels(x, y_arrays, angles=None, values=None, err=10,
                        titles=None, ylabels=None, colors=None,
                        figsize=(16, 9), xlim=(0, 360)):
    """
    Альтернативная версия с явным списком y массивов
    """
    return plot_three_panels_advanced(x, *y_arrays, angles=angles, values=values, 
                                    err=err, titles=titles, ylabels=ylabels, 
                                    colors=colors, figsize=figsize, xlim=xlim)


def slice_r_grad_median(grad, radius_map, angle_map, R=5, bound=0.1, med_deg=1, plot=True):
    # Создаем маску за одну операцию (векторизованно)
    mask = (radius_map > (R - bound/2)) & (radius_map < (R + bound/2))
    
    # Извлекаем значения по маске
    grad_1d = grad[mask]
    phase_1d = angle_map[mask]
    
    # Округляем до ближайшего четного числа (диапазон med_deg градуса)
    phase_rounded = np.round(phase_1d / med_deg).astype(int) * med_deg

    
    # Находим уникальные углы
    unique_phases = np.unique(phase_rounded)
    
    # Вычисляем медианное значение градиента для каждого угла
    grad_median = []
    phase_new = []
    
    for phase in unique_phases:
        # Маска для текущего угла
        phase_mask = (phase_rounded == phase)
        
        # Если есть точки с этим углом, вычисляем медиану
        if np.any(phase_mask):
            grad_median.append(np.median(grad_1d[phase_mask]))
            phase_new.append(phase)

    # Создаем маску для отрицательных медианных градиентов
    negative_mask = np.array(grad_median) < 0
    
    return np.array(grad_median), np.array(phase_new)
    

def slice_r_grad(arr, radius_map, angle_map, R=5, bound=0.1):
    # Создаем маску за одну операцию (векторизованно)
    mask = (radius_map > (R - bound/2)) & (radius_map < (R + bound/2))
    
    # Извлекаем значения по маске
    grad_1d = arr[mask]
    phase_1d = angle_map[mask]
    
    # Создаем маску для отрицательных градиентов
    negative_mask = grad_1d < 0
        
    return grad_1d, phase_1d


def calc_grad_xy(file, dist, telescop="VLT_MUSE", sigma_file=None, plot=True, zoom=False, save=False, shift_vel=0):

    with fits.open(file, memmap=False) as hdul:
        hdu_data = hdul[0].data.copy()
        hdu_header = hdul[0].header.copy()

    if telescop == "VLT_MUSE":
        CD = hdu_header["CD2_2"]
    elif telescop == "ALMA":
        CD = hdu_header["CDELT2"]
    pix_pc = dist*10**(6)*np.pi/180* CD
    
    # Convert angular scale to physical scale [pc/arcsec]
    scale = np.pi * dist * 1000 / 180 / 3600
    vel_grad = np.zeros_like(hdu_data)
    
    for i in range(hdu_data.shape[0]-1):
        for j in range(hdu_data.shape[1]-1):
            vel_grad[i, j] = 1/2*np.sqrt((np.abs(hdu_data[i,j]-hdu_data[i,j+1]) + np.abs(hdu_data[i,j]-hdu_data[i,j-1]))**2 + (np.abs(hdu_data[i,j]-hdu_data[i+1,j]) + np.abs(hdu_data[i,j]-hdu_data[i+1,j]))**2)

    if sigma_file is not None:
        with fits.open(sigma_file) as hdul_sigma:
            sigma_data = hdul_sigma[0].data.copy()  # СОХРАНЯЕМ КОПИЮ ДАННЫХ
            print(f"shape of sigma {sigma_data.shape}")  # Используем КОПИЮ
        vel_grad_xy = np.where(np.abs(vel_grad) > sigma_data, vel_grad, np.nan)
        # vel_grad_xy = np.where(sigma_data < 100, vel_grad_xy, np.nan)
    
    else:
        vel_grad_xy = vel_grad


    if plot:
        create_multiple_plots(hdu_data, vel_grad_xy/pix_pc, titles=["vel", r"$\nabla V_{xy}$"], cmaps="viridis", colorbar_labels=['km/s', "km/s/pc"], norms=["linear", "log"])


    if zoom:
        plot_with_zoom(vel_grad_xy/pix_pc, (400, 350, 650, 550), norm_type="linear", title=r"$V_{xy}$")

    if save:
        import re
        import os
        
        # Извлекаем имя галактики
        match = re.search(r"(NGC\d+)", file)
        if match:
            galaxy_name = match.group(1)
        else:
            galaxy_name = "galaxy"
        
        # Создаем папку если не существует
        os.makedirs(f"data/{galaxy_name}", exist_ok=True)
        
        filename = f"data/{galaxy_name}/{galaxy_name}_{telescop}_vel_grad_xy.fits"
        
        # Используем безопасную запись
        success = safe_fits_write(filename, vel_grad_xy / pix_pc, hdu_header)
        
        if success:
            print(f" Файл сохранен: {filename}")
        else:
            print(f"Не удалось сохранить: {filename}")
    
    return vel_grad_xy / pix_pc 


def calc_grad_r_phi(file_v, file_d, file_a, telescop, dist, PA, size=3, sigma_file=None, plot=True, save=False, slice=True):
    """
    Вычисляет градиенты скорости на основе разностей в подмассивах ixi.
    
    Parameters:
    -----------
    velocity : numpy.ndarray (n x m)
        Массив скоростей.
    distance : numpy.ndarray (n x m)
        Массив расстояний.
    angle : numpy.ndarray (n x m)
        Массив углов.
    
    Returns:
    --------
    grad_r : numpy.ndarray (n x m)
        Радиальный градиент (разность скоростей в точках max и min расстояния).
    grad_phi : numpy.ndarray (n x m)
        Азимутальный градиент (разность скоростей в точках max и min угла).
    """
    
    files = [file_v, file_d, file_a]
    velocity, distance, angle = [fits.open(get_pkg_data_filename(f), memmap=False)[0].data for f in files]
    # n, m = velocity.shape
    head = fits.open(get_pkg_data_filename(file_v), memmap=False)[0].header
    # print(n,m)
    if telescop == "VLT_MUSE":
        CD = head["CD2_2"]
    elif telescop == "ALMA":
        CD = head["CDELT2"]
    pix_pc = dist*10**(6)*np.pi/180* CD

    grad_r = np.zeros_like(velocity)
    grad_phi = np.zeros_like(velocity)
    # print(distance[500:520, 500:520])
    # Проходим по каждому пикселю, кроме граничных (чтобы избежать выхода за границы)
    for i in range(size//2, velocity.shape[0] - size//2 - 1):
        for j in range(size//2, velocity.shape[1] - size//2 - 1):
            # Вырезаем подмассивы 3x3 (и тд)
            vel_patch = velocity[i-size//2:i+1+size//2, j-size//2:j+1+size//2]
            dist_patch = distance[i-size//2:i+size//2 +1, j-size//2:j+1+size//2]
            angle_patch = angle[i-size//2:i+1+size//2, j-size//2:j+1+size//2]
  
            # Находим индексы min и max расстояния
            min_dist_idx = np.unravel_index(np.argmin(dist_patch), dist_patch.shape)
            max_dist_idx = np.unravel_index(np.argmax(dist_patch), dist_patch.shape)
            
            # Находим индексы min и max угла
            min_angle_idx = np.unravel_index(np.argmin(angle_patch), angle_patch.shape)
            max_angle_idx = np.unravel_index(np.argmax(angle_patch), angle_patch.shape)
            
            # Вычисляем разности скоростей
            grad_r[i, j] = vel_patch[max_dist_idx] - vel_patch[min_dist_idx]
            grad_phi[i, j] = vel_patch[max_angle_idx] - vel_patch[min_angle_idx]
    
    grad_phi_ = apply_gradient_sign_by_angle(grad_phi, angle, 90, 270, PA)
    # grad_phi = apply_gradient_sign_by_angle(grad_phi, angle, 270, 360, PA)
    # grad_r = apply_gradient_sign_by_angle(grad_r , angle, 90, 180, PA)
    grad_r_ = apply_gradient_sign_by_angle(grad_r, angle, 180, 360, PA)
    if sigma_file is not None:
        with fits.open(sigma_file) as hdul_sigma:
            sigma_data = hdul_sigma[0].data.copy()  # СОХРАНЯЕМ КОПИЮ ДАННЫХ
            print(f"shape of sigma {sigma_data.shape}")  # Используем КОПИЮ
        grad_phi = np.where(np.abs(grad_phi) > sigma_data, grad_phi_, np.nan)
        # grad_phi = np.where( sigma_data < 100, grad_phi, np.nan)
        grad_r = np.where(np.abs(grad_r) > sigma_data, grad_r_, np.nan)
        # grad_r = np.where( sigma_data < 100, grad_r, np.nan)

    if plot:
        create_multiple_plots(grad_r/pix_pc, grad_phi/pix_pc, vmin_max=[[-1, 1],[-1, 1]], cmaps="viridis", titles=[r"$\nabla V_r$", r"$\nabla V_{\phi}$"], colorbar_labels=['km/s/pc', 'km/s/pc'])
    if slice:
       slice_r_grad(grad_phi, distance, angle)

    if save:
        # Ищем последовательность букв и цифр, которая начинается с NGC
        match = re.search(r"(NGC\d+)", file_v)
        if match:
            galaxy_name = match.group(1)
            print("Galaxy name:", galaxy_name)  # Вывод: {galaxy_name}
            

        head['HISTORY'] = 'Velocities radial gradient calculated'
        head['BUNIT'] = 'km/s/pc'  # Единицы измерения
        
        fits.writeto(f"data/{galaxy_name}/{galaxy_name}_{telescop}_vel_grad_r.fits", data=grad_r/pix_pc, header=head, overwrite=True)
        head['HISTORY'] = 'Velocities azimuthal gradient calculated'
        
        fits.writeto(f"data/{galaxy_name}/{galaxy_name}_{telescop}_vel_grad_phi.fits", data=grad_phi/pix_pc, header=head, overwrite=True)
        print("Files saved.")  

    return grad_r/pix_pc, grad_phi/pix_pc

from scipy.signal import argrelextrema

def safe_fits_write(filename, data, header, max_retries=3):
    """Безопасное сохранение FITS файла с повторными попытками"""
    import time
    import os
    from astropy.io import fits
    
    for attempt in range(max_retries):
        try:
            # Закрываем все открытые FITS файлы
            import gc
            fits_files_closed = 0
            for obj in gc.get_objects():
                if isinstance(obj, fits.HDUList):
                    try:
                        obj.close()
                        fits_files_closed += 1
                    except:
                        pass
            
            if fits_files_closed > 0:
                print(f" Закрыто {fits_files_closed} открытых FITS файлов")
            
            # Создаем папку если нет
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Пытаемся сохранить
            fits.writeto(filename, data=data, header=header, overwrite=True)
            print(f" Файл сохранен: {filename}")
            return True
            
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f" Попытка {attempt + 1}: Файл занят, жду 1 секунду...")
                time.sleep(1)
            else:
                print(f" Не удалось сохранить {filename} после {max_retries} попыток: {e}")
                return False
        except Exception as e:
            print(f"Ошибка сохранения {filename}: {e}")
            return False
        
def find_local_maxima_scipy(x, y, order=1000):
    """
    Поиск локальных максимумов с помощью SciPy
    
    Parameters:
    order: int - сколько точек с каждой стороны сравнивать
    """
    # Находим индексы локальных максимумов
    local_max_indices = argrelextrema(y, np.greater, order=order)[0]
    
    local_max_angles = x[local_max_indices]
    local_max_values = y[local_max_indices]
    
    return local_max_indices, local_max_angles, local_max_values

def azimuthal_scan(Flux, telescop, grad1, grad2, angle_map, distance_map, dist=20.3, R=5, bound=0.01):
    
    files = [Flux, grad1, grad2, angle_map, distance_map]
    flux, grad_r, grad_phi, angle, distance  = [fits.open(get_pkg_data_filename(f), memmap=False)[0].data for f in files]
    if telescop == "VLT_MUSE":
        CD = fits.open(get_pkg_data_filename(grad1))[0].header["CD2_2"]
    elif telescop == "ALMA":
        CD = fits.open(get_pkg_data_filename(grad1))[0].header["CDELT2"]
    pix_pc = dist*10**(6) * np.pi/180 * CD

    print(f"shape of flux {flux.shape}, grad - {grad_r.shape}, {grad_phi.shape}")
    F, x_f= slice_r_grad(flux, distance, angle, R, bound)
    indices, angles, values = find_local_maxima_scipy(x_f, F, order=1000)
    v_phi, x = slice_r_grad(grad_phi*pix_pc, distance, angle, R, bound)
    v_r, _ = slice_r_grad(grad_r*pix_pc, distance, angle, R, bound)
    print(f"len of flux {F.shape}, grad - {v_r.shape}, {v_phi.shape}")

    plot_three_panels_advanced(x, F, v_phi, v_r, angles=angles, values=values, xf=x_f, legend=rf"{R} $\pm$ {bound} ", titles=[r'$Flux$', r'$V_{\phi}$', r'$V_r$'], ylabels=[" ", "km/s", "km/s"], err=np.nan, yscales=["log", "symlog", "symlog"])


    

def run_gradient_analysis(galaxy_name, telescop, dist, velocity_file, distance_file, angle_file, PA, sigma_file=None,
                         size=3, plot=True, save=True, shift_vel=0):
    """
    Run gradient analysis for a galaxy.
    
    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy
    dist : float
        Distance to the galaxy in Mpc
    velocity_file : str
        Path to the velocity field FITS file
    distance_file : str
        Path to the distance map FITS file
    angle_file : str
        Path to the angle map FITS file
    size : int, optional
        Size of the patch for gradient calculation, by default 3
    plot : bool, optional
        Whether to plot results, by default True
    save : bool, optional
        Whether to save results, by default True
    
    Returns
    -------
    grad_xy : numpy.ndarray
        XY gradient map
    grad_r : numpy.ndarray
        Radial gradient map
    grad_phi : numpy.ndarray
        Azimuthal gradient map
    """
    
    # Calculate XY gradient
    grad_xy = calc_grad_xy(velocity_file, dist, sigma_file=sigma_file, telescop=telescop, plot=plot, save=save, shift_vel=shift_vel)
    
    # Calculate radial and azimuthal gradients
    grad_r, grad_phi = calc_grad_r_phi(velocity_file, distance_file, angle_file, telescop, dist, PA, 
                                       size=size, sigma_file=sigma_file, plot=plot, save=save)
    
    return grad_xy, grad_r, grad_phi

def run_azimuthal_analysis(galaxy_name, telescop, dist, flux_file, grad_r_file, grad_phi_file, 
                          angle_file, distance_file, R=5, bound=0.01):
    """
    Run azimuthal scan for a galaxy.
    
    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy
    dist : float
        Distance to the galaxy in Mpc
    flux_file : str
        Path to the flux FITS file
    grad_r_file : str
        Path to the radial gradient FITS file
    grad_phi_file : str
        Path to the azimuthal gradient FITS file
    angle_file : str
        Path to the angle map FITS file
    distance_file : str
        Path to the distance map FITS file
    R : float, optional
        Radius for the slice, by default 5
    bound : float, optional
        Bound for the slice, by default 0.01
    """
    
    azimuthal_scan(flux_file, telescop, grad_r_file, grad_phi_file, angle_file, distance_file, 
                   dist=dist, R=R, bound=bound)
    

def plot_supernovae_on_image(fits_file, sn_file, output_file=None):
    """
    Наносит позиции сверхновых из текстового файла на изображение FITS
    """
    
    # Чтение FITS файла
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        
    # Создание WCS объекта для преобразования координат
    wcs = WCS(header)
    
    # Чтение данных о сверхновых вручную
    sn_data = read_sn_data_manual(sn_file)
    
    print("Прочитанные данные о сверхновых:")
    # print(sn_data)
    # print(f"Типы данных: {sn_data.dtypes}")
    
    # Создание графика
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(projection=wcs)
    
    # Отображение изображения FITS
    if data is not None:

        norm = simple_norm(data, "log", percent=99, log_a=50)
        ax.imshow(data, origin='lower', cmap='viridis', norm=norm)
       
    # Разные цвета для разных типов сверхновых
    type_colors = {
        'II': 'red',
        'IIn': 'orange',
        'Ib': 'green',
        'Ic': 'blue',
        'Ia': 'purple'
    }
    
    # Нанесение позиций сверхновых
    plotted_labels = set()
    
    for idx, sn in sn_data.iterrows():
        try:
            # Преобразуем в числа, убеждаясь что это строки
            ra_str = str(sn['R.A.']).strip()
            dec_str = str(sn['Dec']).strip()
            
            # Убираем возможные лишние символы
            ra_str = ra_str.replace(' ', '')
            dec_str = dec_str.replace(' ', '')
            
            ra = float(ra_str)
            dec = float(dec_str)
            
            sn_type = str(sn['Type']).strip()
            sn_name = str(sn['Supernova']).strip()
            
            print(f"Обрабатываю {sn_name}: RA={ra} (тип: {type(ra)}), Dec={dec} (тип: {type(dec)}), Type={sn_type}")
            
            # Преобразование RA/DEC в пиксельные координаты
            # Убеждаемся, что это числа с плавающей точкой
            world_coords = np.array([[ra, dec]], dtype=np.float64)
            # print(f"World coordinates: {world_coords}, dtype: {world_coords.dtype}")
            
            pixel_coords = wcs.wcs_world2pix(world_coords, 0)
            # print(f"Pixel coordinates: {pixel_coords}")
            
            # Выбор цвета в зависимости от типа
            color = type_colors.get(sn_type, 'yellow')
            
            # Создание метки для легенды
            label = f'{sn_type}'
            legend_label = label if label not in plotted_labels else ""
            plotted_labels.add(label)
            
            # Нанесение маркера
            ax.scatter(pixel_coords[0, 0], pixel_coords[0, 1], 
                      color=color, s=50, marker='*', edgecolors='white', 
                      linewidth=1, label=legend_label, zorder=5, alpha=0.5)
            
            # Добавление текста с названием
            ax.text(pixel_coords[0, 0] + 15, pixel_coords[0, 1] + 15, 
                   sn_name, color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                   zorder=6)
            
        except Exception as e:
            print(f"Ошибка при обработке сверхновой {sn}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Настройка осей
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('Dec (J2000)')
    
    # Легенда
    if plotted_labels:
        ax.legend(loc='upper right', framealpha=0.8, title='Supernova Types')
    
    plt.title(f'Supernovae positions on {fits_file.split("/")[-1]}')
    
    # Сохранение или отображение
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Изображение сохранено как: {output_file}")
    else:
        plt.show()
    
    plt.close()

def read_sn_data_manual(sn_file):
    """Ручное чтение файла со сверхновыми с улучшенной обработкой"""
    data = []
    with open(sn_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.strip()
        # Пропускаем заголовок и пустые строки
        if not line or line.startswith('Galaxy') or line.startswith('---'):
            continue
            
        # print(f"Обрабатываю строку {i}: {line}")
        
        # Разделяем по пробелам и фильтруем пустые элементы
        parts = [p for p in line.split(' ') if p]
        # print(f"Разделенные части: {parts}")
        
        if len(parts) >= 5:
            # Обрабатываем специальные символы в галактике
            galaxy = parts[0]
            supernova = parts[1]
            sn_type = parts[2]
            
            # Координаты могут быть в разных позициях, ищем числа
            ra = None
            dec = None
            for j in range(3, len(parts)):
                if ra is None and is_float(parts[j]):
                    ra = parts[j]
                elif ra is not None and dec is None and is_float(parts[j]):
                    dec = parts[j]
                    break
            
            if ra is not None and dec is not None:
                # Остальные поля
                insample = parts[j+1] if j+1 < len(parts) else ''
                reference = ' '.join(parts[j+2:]) if j+2 < len(parts) else ''
                
                data.append([galaxy, supernova, sn_type, ra, dec, insample, reference])
    
    columns = ['Galaxy', 'Supernova', 'Type', 'R.A.', 'Dec', 'InSample', 'Reference']
    df = pd.DataFrame(data, columns=columns)
    
    # Явно преобразуем координаты в числа
    df['R.A.'] = pd.to_numeric(df['R.A.'], errors='coerce')
    df['Dec'] = pd.to_numeric(df['Dec'], errors='coerce')
    
    # Удаляем строки с некорректными координатами
    df = df.dropna(subset=['R.A.', 'Dec'])
    
    print("Создан DataFrame:")
    print(df)
    print(f"Типы колонок: {df.dtypes}")
    
    return df

def is_float(value):
    """Проверяет, можно ли преобразовать строку в float"""
    try:
        float(value)
        return True
    except ValueError:
        return False


def plot_supernovae_on_image_zoom(fits_file, sn_file, output_file=None, zoom_size=100):
    """
    Наносит позиции сверхновых из текстового файла на изображение FITS
    и создает зуммированные изображения вокруг каждой сверхновой
    
    Parameters:
    -----------
    fits_file : str
        Путь к FITS файлу с изображением
    sn_file : str
        Путь к текстовому файлу с данными о сверхновых
    output_file : str, optional
        Путь для сохранения результата (если None, показывается на экране)
    zoom_size : int
        Размер зуммированной области в пикселях (квадрат zoom_size x zoom_size)
    """
    
    # Чтение FITS файла
    with fits.open(fits_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        
    # Создание WCS объекта для преобразования координат
    wcs = WCS(header)
    
    # Чтение данных о сверхновых вручную
    sn_data = read_sn_data_manual(sn_file)
    
    print("Прочитанные данные о сверхновых:")
    print(sn_data)
    
    # Создание основной карты
    fig_main = plt.figure(figsize=(15, 12))
    ax_main = plt.subplot(projection=wcs)
    
    # Отображение изображения FITS
    if data is not None:
        data_clean = data[~np.isnan(data)]
        if len(data_clean) > 0:
            vmin = np.percentile(data_clean, 10)
            vmax = np.percentile(data_clean, 99)
            ax_main.imshow(data, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
        else:
            ax_main.imshow(data, origin='lower', cmap='gray')
    
    # Разные цвета для разных типов сверхновых
    type_colors = {
        'II': 'red',
        'Ia': 'blue',
        'Ib': 'green',
        'Ic': 'orange',
        'IIn': 'purple'
    }
    
    # Нанесение позиций сверхновых на основную карту
    plotted_labels = set()
    
    for idx, sn in sn_data.iterrows():
        try:
            ra = float(sn['R.A.'])
            dec = float(sn['Dec'])
            sn_type = str(sn['Type']).strip()
            sn_name = str(sn['Supernova']).strip()
            
            # Преобразование RA/DEC в пиксельные координаты
            world_coords = np.array([[ra, dec]], dtype=np.float64)
            pixel_coords = wcs.wcs_world2pix(world_coords, 0)
            x, y = pixel_coords[0]
            
            print(f"Обрабатываю {sn_name}: X={x:.1f}, Y={y:.1f}")
            
            # Выбор цвета в зависимости от типа
            color = type_colors.get(sn_type, 'yellow')
            
            # Создание метки для легенды
            label = f'{sn_type}'
            legend_label = label if label not in plotted_labels else ""
            plotted_labels.add(label)
            
            # Нанесение маркера на основную карту
            ax_main.scatter(x, y, color=color, s=350, marker='*', 
                          edgecolors='white', linewidth=1, 
                          label=legend_label, zorder=5, alpha=0.8)
            
            # Добавление текста с названием на основную карту
            ax_main.text(x + 15, y + 15, sn_name, color=color, 
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                       zorder=6)
            
            # Создание зуммированного изображения для каждой сверхновой
            create_zoom_image(data, x, y, sn_name, sn_type, wcs, zoom_size, color)
            
        except Exception as e:
            print(f"Ошибка при обработке сверхновой {sn}: {e}")
            continue
    
    # Настройка основной карты
    ax_main.set_xlabel('RA (J2000)')
    ax_main.set_ylabel('Dec (J2000)')
    
    # Легенда
    if plotted_labels:
        ax_main.legend(loc='upper right', framealpha=0.8, title='Supernova Types')
    
    ax_main.set_title(f'Supernovae positions on {os.path.basename(fits_file)}')
    
    # Сохранение или отображение основной карты
    if output_file:
        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Основная карта сохранена как: {output_file}")
    else:
        plt.show()
    
    plt.close(fig_main)

def create_zoom_image(data, x, y, sn_name, sn_type, wcs, zoom_size, color):
    """
    Создает зуммированное изображение вокруг позиции сверхновой
    """
    # Преобразуем координаты в целые числа
    x_int = int(round(x))
    y_int = int(round(y))
    
    # Определяем границы зуммированной области
    half_size = zoom_size // 2
    x_min = max(0, x_int - half_size)
    x_max = min(data.shape[1], x_int + half_size)
    y_min = max(0, y_int - half_size)
    y_max = min(data.shape[0], y_int + half_size)
    
    # Вырезаем область
    zoom_data = data[y_min:y_max, x_min:x_max]
    
    if zoom_data.size == 0:
        print(f"Предупреждение: невозможно вырезать область для {sn_name}")
        return
    
    # Создаем WCS для зуммированной области
    zoom_wcs = wcs[y_min:y_max, x_min:x_max]
    
    # Создаем фигуру для зуммированного изображения
    fig_zoom = plt.figure(figsize=(8, 8))
    ax_zoom = plt.subplot(projection=zoom_wcs)
    
    # Отображаем зуммированную область
    if zoom_data is not None:
        norm = simple_norm(zoom_data, "log", percent=99, log_a=50)
        ax_zoom.imshow(zoom_data, origin='lower', cmap='viridis', norm=norm)
    
    # Отмечаем позицию сверхновой в зуммированной области
    zoom_x = x - x_min
    zoom_y = y - y_min
    
    ax_zoom.scatter(zoom_x, zoom_y, color=color, s=200, marker='*',
                   edgecolors='white', linewidth=2, zorder=5, alpha=0.9)
    
    # Добавляем крест для точного позиционирования
    ax_zoom.axvline(x=zoom_x, color=color, linestyle='--', alpha=0.5, linewidth=1)
    ax_zoom.axhline(y=zoom_y, color=color, linestyle='--', alpha=0.5, linewidth=1)
    
    # Настройка зуммированного изображения
    ax_zoom.set_xlabel('RA (J2000)')
    ax_zoom.set_ylabel('Dec (J2000)')
    ax_zoom.set_title(f'{sn_name} ({sn_type})\nZoom: {zoom_size}x{zoom_size} pixels')
    
    # Добавляем координаты
    ra_dec = zoom_wcs.wcs_pix2world([[zoom_x, zoom_y]], 0)[0]
    ax_zoom.text(0.05, 0.95, f'RA: {ra_dec[0]:.4f}°\nDec: {ra_dec[1]:.4f}°',
                transform=ax_zoom.transAxes, color=color, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                verticalalignment='top')
    
    # Сохраняем зуммированное изображение
    zoom_output_dir = "pic/trash/zoom_regions/"
    os.makedirs(zoom_output_dir, exist_ok=True)
    zoom_filename = f"{zoom_output_dir}{sn_name}_zoom_{zoom_size}px.png"
    plt.savefig(zoom_filename, dpi=200, bbox_inches='tight')
    print(f"Зуммированная область сохранена: {zoom_filename}")
    
    plt.close(fig_zoom)

