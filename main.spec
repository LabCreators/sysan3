# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.py', '__init__.py', 'arima.py', 'basis_gen.py', 'depict_poly.py', 'bruteforce.py', 'bruteforce_window.py', 'calculate_optimal_degrees.py', 'main_window.py', 'new_main.py', 'output.py', 'parser.py', 'solve.py', 'syst_solution.py', 'task_solution.py', 'task_solution_custom.py'],
             pathex=['C:\\Users\\Roman\\PycharmProjects\\sysan3_new'],
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
