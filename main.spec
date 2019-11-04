# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.py', '__init__.py', 'arima.py', 'basis_gen.py', 'bruteforce.py', 'bruteforce_window.py', 'calculate_optimal_degrees.py', 'depict_poly.py', 'main_window.py', 'output.py', 'solve.py', 'syst_solution.py', 'task_solution.py'],
             pathex=['D:\\КПИ им. Сикорского\\Системный анализ\\Smirnov_Lab3\\sysan3'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
