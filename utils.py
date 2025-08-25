import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def create_variable_boxplot(variable_index, df_list, figsize=(15, 8)):
    """
    Crea un boxplot para una variable específica de todas las estaciones
    
    Parameters:
    variable_index (int): Índice de la variable (1-15, ya que 0 es Date)
    figsize (tuple): Tamaño de la figura
    """
    plt.figure(figsize=figsize)
    
    # Preparar datos para el boxplot
    data_for_boxplot = []
    station_labels = []
    variable_names = []

    # Obtener la variable especificada de cada estación
    for station_name, station_df in df_list.items():
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index].dropna()
            data_for_boxplot.append(variable_data)
            station_labels.append(station_name)
            variable_names.append(station_df.columns[variable_index])
    
    # Crear el boxplot
    plt.boxplot(data_for_boxplot, tick_labels=station_labels)
    plt.xticks(rotation=45, ha='right')
    
    # Obtener el nombre común de la variable (sin el prefijo de estación)
    if variable_names:
        # Extraer la parte común del nombre de la variable
        common_var_name = variable_names[0].split(' ', 1)[1] if ' ' in variable_names[0] else variable_names[0]
        plt.title(f'Boxplot de {common_var_name} por Estación', fontsize=14, fontweight='bold')
        
        # Determinar la unidad de medida para el label del eje Y
        if 'ppm' in common_var_name:
            ylabel = 'Concentración (ppm)'
        elif 'ppb' in common_var_name:
            ylabel = 'Concentración (ppb)'
        elif 'ug/m3' in common_var_name:
            ylabel = 'Concentración (μg/m³)'
        elif 'mmhg' in common_var_name:
            ylabel = 'Presión (mmHg)'
        elif 'mm/hr' in common_var_name:
            ylabel = 'Precipitación (mm/hr)'
        elif '%' in common_var_name:
            ylabel = 'Porcentaje (%)'
        elif 'KW/m2' in common_var_name:
            ylabel = 'Radiación Solar (KW/m²)'
        elif 'degC' in common_var_name:
            ylabel = 'Temperatura (°C)'
        elif 'KMPH' in common_var_name:
            ylabel = 'Velocidad del Viento (km/h)'
        elif 'DEG' in common_var_name:
            ylabel = 'Dirección del Viento (grados)'
        else:
            ylabel = 'Valor'
            
        plt.ylabel(ylabel)
    
    plt.xlabel('Estaciones')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas básicas
    print(f"Variable {variable_index}: {common_var_name if variable_names else 'Desconocida'}")
    print("-" * 60)
    for i, (station_name, station_df) in enumerate(df_list.items()):
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index].dropna()
            print(f"{station_name}: Media={variable_data.mean():.2f}, Mediana={variable_data.median():.2f}, "
                  f"Desv.Est={variable_data.std():.2f}")

def create_correlation_heatmap(variable_index, df_list,figsize=(12, 10)):
    """
    Crea un mapa de correlación (heatmap) para una variable específica entre todas las estaciones
    
    Parameters:
    variable_index (int): Índice de la variable (1-15, ya que 0 es Date)
    figsize (tuple): Tamaño de la figura
    """
    
    # Crear un DataFrame con todas las estaciones para la variable específica
    correlation_data = pd.DataFrame()
    variable_name = None

    for station_name, station_df in df_list.items():
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index]
            correlation_data[station_name] = variable_data
            if variable_name is None:
                variable_name = station_df.columns[variable_index]
    
    # Calcular la matriz de correlación
    correlation_matrix = correlation_data.corr()
    
    # Crear el heatmap
    plt.figure(figsize=figsize)
    
    # Usar seaborn para crear un heatmap más atractivo
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # Máscara para mostrar solo la mitad inferior
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdYlBu_r', 
                center=0,
                fmt='.2f',
                square=True,
                annot_kws={'size': 8},
                cbar_kws={"shrink": .8})
    
    # Obtener el nombre común de la variable
    common_var_name = variable_name.split(' ', 1)[1] if ' ' in variable_name else variable_name
    
    plt.title(f'Matriz de Correlación - {common_var_name}\nCorrelaciones entre Estaciones', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Imprimir estadísticas de correlación
    print(f"Variable {variable_index}: {common_var_name}")
    print("-" * 60)
    
    # Encontrar las correlaciones más altas y más bajas (excluyendo la diagonal)
    correlation_values = correlation_matrix.values.copy()
    np.fill_diagonal(correlation_values, np.nan)  # Eliminar la diagonal
    
    # Obtener índices de correlación máxima y mínima
    max_corr_idx = np.unravel_index(np.nanargmax(correlation_values), correlation_values.shape)
    min_corr_idx = np.unravel_index(np.nanargmin(correlation_values), correlation_values.shape)
    
    max_corr_value = correlation_values[max_corr_idx]
    min_corr_value = correlation_values[min_corr_idx]
    
    stations_list = list(correlation_matrix.columns)
    
    print(f"Correlación más alta: {stations_list[max_corr_idx[0]]} - {stations_list[max_corr_idx[1]]}: {max_corr_value:.3f}")
    print(f"Correlación más baja: {stations_list[min_corr_idx[0]]} - {stations_list[min_corr_idx[1]]}: {min_corr_value:.3f}")
    
    # Estadísticas generales de correlación
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    correlations = upper_triangle.stack().values
    
    print(f"Correlación promedio entre estaciones: {np.mean(correlations):.3f}")
    print(f"Desviación estándar de correlaciones: {np.std(correlations):.3f}")
    print(f"Correlaciones > 0.8: {np.sum(correlations > 0.8)} de {len(correlations)} pares")
    print(f"Correlaciones < 0.5: {np.sum(correlations < 0.5)} de {len(correlations)} pares")

def generate_correction_report(correction_summary, dfs_original, dfs_corrected):
    """
    Genera un reporte detallado de las correcciones realizadas
    """
    print("\n" + "="*80)
    print("REPORTE DETALLADO DE CORRECCIONES")
    print("="*80)
    
    total_corrections = 0
    stations_with_corrections = 0
    
    for station_name, corrections in correction_summary.items():
        if corrections:  # Si hay correcciones en esta estación
            stations_with_corrections += 1
            station_total = sum([corr['negative_count'] for corr in corrections.values()])
            total_corrections += station_total
            
            print(f"\n📍 ESTACIÓN: {station_name}")
            print(f"   Total de valores corregidos: {station_total}")
            print("   Variables afectadas:")
            
            for var_name, corr_info in corrections.items():
                print(f"     • {var_name}:")
                print(f"       - Valores negativos: {corr_info['negative_count']}")
                print(f"       - Rango original: [{corr_info['original_min']:.3f}, {corr_info['original_max']:.3f}]")
                print(f"       - Rango corregido: [{corr_info['corrected_min']:.3f}, {corr_info['corrected_max']:.3f}]")
        else:
            print(f"\n✅ ESTACIÓN: {station_name} - Sin valores negativos")
    
    print(f"\n" + "="*80)
    print("RESUMEN GENERAL:")
    print(f"• Total de estaciones procesadas: {len(correction_summary)}")
    print(f"• Estaciones con correcciones: {stations_with_corrections}")
    print(f"• Total de valores negativos corregidos: {total_corrections}")
    print("="*80)

def validate_corrections(dfs_original, dfs_corrected):
    """
    Valida que las correcciones se realizaron correctamente
    """
    print("\n🔍 VALIDACIÓN DE CORRECCIONES:")
    print("-" * 40)
    
    for station_name in dfs_original.keys():
        df_orig = dfs_original[station_name]
        df_corr = dfs_corrected[station_name]
        
        # Verificar que no haya valores negativos en datos corregidos
        negative_remaining = 0
        for col_idx in range(1, len(df_corr.columns)):
            negative_remaining += (df_corr.iloc[:, col_idx] < 0).sum()
        
        if negative_remaining == 0:
            print(f"✅ {station_name}: Todos los valores negativos fueron corregidos")
        else:
            print(f"❌ {station_name}: Aún quedan {negative_remaining} valores negativos")

def detect_and_fix_negative_values(dfs_dict, method='linear', verbose=True):
    """
    Detecta y corrige valores negativos en todas las variables de todas las estaciones
    usando interpolación con valores no negativos circundantes.
    
    Parameters:
    dfs_dict (dict): Diccionario con DataFrames de estaciones
    method (str): Método de interpolación ('linear', 'polynomial', 'spline')
    verbose (bool): Si mostrar información detallada del proceso
    
    Returns:
    dict: Diccionario con DataFrames corregidos
    dict: Resumen de correcciones realizadas
    """
    
    corrected_dfs = {}
    correction_summary = {}
    
    print("DETECCIÓN Y CORRECCIÓN DE VALORES NEGATIVOS")
    print("=" * 60)
    
    for station_name, df in dfs_dict.items():
        print(f"\nProcesando estación: {station_name}")
        print("-" * 40)
        
        # Crear copia del DataFrame
        df_corrected = df.copy()
        station_corrections = {}
        
        # Procesar cada variable (excluyendo la columna Date)
        for col_idx in range(1, len(df.columns)):  # Empezar desde 1 para saltar Date
            col_name = df.columns[col_idx]
            variable_data = df_corrected.iloc[:, col_idx].copy()
            
            # Detectar valores negativos
            negative_mask = variable_data < 0
            negative_count = negative_mask.sum()
            less_than_minus30 = variable_data < -30
            if col_idx == 13:
                negative_count = less_than_minus30.sum()

            if negative_count > 0:
                if verbose:
                    print(f"  Variable {col_idx} ({col_name}): {negative_count} valores negativos encontrados")
                
                # Crear serie para interpolación
                series_for_interp = variable_data.copy()
                
                # Reemplazar valores negativos con NaN para interpolación
                series_for_interp[negative_mask] = np.nan
                
                # Realizar interpolación
                if method == 'linear':
                    interpolated = series_for_interp.interpolate(method='linear')
                elif method == 'polynomial':
                    interpolated = series_for_interp.interpolate(method='polynomial', order=2)
                elif method == 'spline':
                    interpolated = series_for_interp.interpolate(method='spline', order=2)
                else:
                    interpolated = series_for_interp.interpolate(method='linear')
                
                # Si aún hay NaN al principio o final, usar forward/backward fill
                interpolated = interpolated.fillna(method='bfill').fillna(method='ffill')
                
                # Si todavía hay NaN, usar la media de los valores positivos
                if interpolated.isna().any():
                    positive_mean = variable_data[variable_data >= 0].mean()
                    if not np.isnan(positive_mean):
                        interpolated = interpolated.fillna(positive_mean)
                    else:
                        interpolated = interpolated.fillna(0)
                
                # Reemplazar solo los valores que eran negativos
                df_corrected.iloc[negative_mask, col_idx] = interpolated[negative_mask]
                
                # Guardar estadísticas de corrección
                original_negative_values = variable_data[negative_mask]
                corrected_values = interpolated[negative_mask]
                
                station_corrections[col_name] = {
                    'negative_count': negative_count,
                    'original_min': original_negative_values.min(),
                    'original_max': original_negative_values.max(),
                    'corrected_min': corrected_values.min(),
                    'corrected_max': corrected_values.max(),
                    'corrected_mean': corrected_values.mean()
                }
                
                if verbose:
                    print(f"    -> Corregidos: min={corrected_values.min():.3f}, "
                          f"max={corrected_values.max():.3f}, mean={corrected_values.mean():.3f}")
            
            else:
                if verbose:
                    print(f"  Variable {col_idx} ({col_name}): Sin valores negativos")
        
        corrected_dfs[station_name] = df_corrected
        correction_summary[station_name] = station_corrections
    
    return corrected_dfs, correction_summary

def compare_datasets_statistics(dfs_original, dfs_corrected, variable_index):
    """
    Compara estadísticas entre datasets originales y corregidos para una variable específica
    """
    print(f"\n📊 COMPARACIÓN ESTADÍSTICA - Variable {variable_index}")
    print("="*70)
    
    comparison_stats = []
    
    for station_name in dfs_original.keys():
        df_orig = dfs_original[station_name]
        df_corr = dfs_corrected[station_name]
        
        if variable_index < len(df_orig.columns):
            var_name = df_orig.columns[variable_index]
            
            orig_data = df_orig.iloc[:, variable_index].dropna()
            corr_data = df_corr.iloc[:, variable_index].dropna()
            
            # Calcular estadísticas
            stats_comparison = {
                'Station': station_name,
                'Variable': var_name,
                'Original_Mean': orig_data.mean(),
                'Corrected_Mean': corr_data.mean(),
                'Original_Std': orig_data.std(),
                'Corrected_Std': corr_data.std(),
                'Original_Min': orig_data.min(),
                'Corrected_Min': corr_data.min(),
                'Original_Negatives': (orig_data < 0).sum(),
                'Corrected_Negatives': (corr_data < 0).sum()
            }
            
            comparison_stats.append(stats_comparison)
            
            # Mostrar comparación por estación
            print(f"\n🏭 {station_name}:")
            print(f"   Media: {orig_data.mean():.3f} → {corr_data.mean():.3f}")
            print(f"   Mínimo: {orig_data.min():.3f} → {corr_data.min():.3f}")
            print(f"   Negativos: {(orig_data < 0).sum()} → {(corr_data < 0).sum()}")
    
    return pd.DataFrame(comparison_stats)

def create_before_after_plots(dfs_original, dfs_corrected, variable_index, station_name):
    """
    Crea gráficos de comparación antes y después para una variable y estación específica
    """
    df_orig = dfs_original[station_name]
    df_corr = dfs_corrected[station_name]
    
    if variable_index < len(df_orig.columns):
        var_name = df_orig.columns[variable_index]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        orig_data = df_orig.iloc[:, variable_index]
        corr_data = df_corr.iloc[:, variable_index]
        
        # Serie temporal - Original
        axes[0, 0].plot(orig_data, alpha=0.7, color='red', label='Original')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 0].set_title(f'Serie Temporal Original - {station_name}')
        axes[0, 0].set_ylabel('Valor')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Serie temporal - Corregido
        axes[0, 1].plot(corr_data, alpha=0.7, color='green', label='Corregido')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_title(f'Serie Temporal Corregida - {station_name}')
        axes[0, 1].set_ylabel('Valor')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histograma comparativo
        axes[1, 0].hist(orig_data.dropna(), bins=30, alpha=0.6, color='red', label='Original')
        axes[1, 0].hist(corr_data.dropna(), bins=30, alpha=0.6, color='green', label='Corregido')
        axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Distribución Comparativa')
        axes[1, 0].set_xlabel('Valor')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Boxplot comparativo
        axes[1, 1].boxplot([orig_data.dropna(), corr_data.dropna()], 
                          labels=['Original', 'Corregido'])
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Boxplot Comparativo')
        axes[1, 1].set_ylabel('Valor')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Análisis Antes/Después - Variable {variable_index}: {var_name}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Estadísticas detalladas
        print(f"\n📈 ESTADÍSTICAS DETALLADAS:")
        print(f"Variable: {var_name}")
        print(f"Estación: {station_name}")
        print("-" * 50)
        print("ORIGINAL:")
        print(f"  Media: {orig_data.mean():.4f}")
        print(f"  Mediana: {orig_data.median():.4f}")
        print(f"  Desv. Est.: {orig_data.std():.4f}")
        print(f"  Min: {orig_data.min():.4f}")
        print(f"  Max: {orig_data.max():.4f}")
        print(f"  Valores negativos: {(orig_data < 0).sum()}")
        print(f"  Valores faltantes: {orig_data.isna().sum()}")
        
        print("\nCORREGIDO:")
        print(f"  Media: {corr_data.mean():.4f}")
        print(f"  Mediana: {corr_data.median():.4f}")
        print(f"  Desv. Est.: {corr_data.std():.4f}")
        print(f"  Min: {corr_data.min():.4f}")
        print(f"  Max: {corr_data.max():.4f}")
        print(f"  Valores negativos: {(corr_data < 0).sum()}")
        print(f"  Valores faltantes: {corr_data.isna().sum()}")

def create_compact_histogram(variable_index, df_list, figsize=(16, 12)):
    """
    Crea histogramas compactos para una variable específica de todas las estaciones
    con pruebas de normalidad
    
    Parameters:
    variable_index (int): Índice de la variable (1-15, ya que 0 es Date)
    figsize (tuple): Tamaño de la figura
    """
    
    # Preparar datos
    data_by_station = {}
    variable_name = None

    for station_name, station_df in df_list.items():
        if variable_index < len(station_df.columns):
            variable_data = station_df.iloc[:, variable_index].dropna()
            data_by_station[station_name] = variable_data
            if variable_name is None:
                variable_name = station_df.columns[variable_index]
    
    # Obtener el nombre común de la variable
    common_var_name = variable_name.split(' ', 1)[1] if ' ' in variable_name else variable_name
    
    # Crear subplot con 3 filas y 5 columnas para 15 estaciones
    fig, axes = plt.subplots(3, 5, figsize=figsize)
    axes = axes.flatten()
    
    # Lista para almacenar resultados de normalidad
    normality_results = []
    
    for i, (station_name, data) in enumerate(data_by_station.items()):
        ax = axes[i]
        
        # Crear histograma
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Añadir curva normal teórica
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal teórica')
        
        # Prueba de normalidad (Shapiro-Wilk para muestras pequeñas, Anderson-Darling para grandes)
        if len(data) <= 5000:
            stat, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        else:
            # Para muestras muy grandes, usar Kolmogorov-Smirnov
            stat, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
            test_name = "Kolmogorov-Smirnov"
        
        # Determinar si es normal (p > 0.05)
        is_normal = p_value > 0.05
        color = 'green' if is_normal else 'red'
        
        # Título con información de normalidad
        ax.set_title(f'{station_name}\np={p_value:.4f}', fontsize=10, color=color, fontweight='bold')
        ax.set_xlabel('Valor', fontsize=8)
        ax.set_ylabel('Densidad', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Guardar resultado
        normality_results.append({
            'Station': station_name,
            'Test': test_name,
            'Statistic': stat,
            'P-value': p_value,
            'Is_Normal': is_normal,
            'Mean': mu,
            'Std': sigma,
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data)
        })
    
    # Título general
    plt.suptitle(f'Histogramas de {common_var_name} por Estación\n(Verde: Normal, Rojo: No Normal)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Crear resumen estadístico
    results_df = pd.DataFrame(normality_results)
    
    print(f"Variable {variable_index}: {common_var_name}")
    print("=" * 80)
    print(f"Estaciones con distribución NORMAL (p > 0.05): {results_df['Is_Normal'].sum()} de {len(results_df)}")
    print(f"Estaciones con distribución NO NORMAL (p ≤ 0.05): {(~results_df['Is_Normal']).sum()} de {len(results_df)}")
    
    print(f"\nP-valores promedio por prueba:")
    for test in results_df['Test'].unique():
        test_results = results_df[results_df['Test'] == test]
        print(f"  {test}: {test_results['P-value'].mean():.4f} (n={len(test_results)})")
    
    print(f"\nEstaciones con distribución MÁS NORMAL (p-valor más alto):")
    top_normal = results_df.nlargest(3, 'P-value')[['Station', 'P-value', 'Skewness', 'Kurtosis']]
    for _, row in top_normal.iterrows():
        print(f"  {row['Station']}: p={row['P-value']:.4f}, Asimetría={row['Skewness']:.3f}, Curtosis={row['Kurtosis']:.3f}")
    
    print(f"\nEstaciones con distribución MENOS NORMAL (p-valor más bajo):")
    bottom_normal = results_df.nsmallest(3, 'P-value')[['Station', 'P-value', 'Skewness', 'Kurtosis']]
    for _, row in bottom_normal.iterrows():
        print(f"  {row['Station']}: p={row['P-value']:.4f}, Asimetría={row['Skewness']:.3f}, Curtosis={row['Kurtosis']:.3f}")
    
    return results_df








