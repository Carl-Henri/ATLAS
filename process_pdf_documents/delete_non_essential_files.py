import os
import shutil

# Fonction pour supprimer les fichiers inutiles et lourds (mais utiles pour apprécier la qualité du parsing)
def delete_non_essential_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Suppression des dossiers "pictures_deleted" et "tables_serialized" dans dirnames
        for dirname in dirnames[:]:
            if dirname in ("pictures_deleted", "tables_serialized"):
                dir_to_delete = os.path.join(dirpath, dirname)
                print(f"Suppression du dossier : {dir_to_delete}")
                shutil.rmtree(dir_to_delete)
                # On enlève le dossier de la liste pour éviter que os.walk n'essaie de le parcourir
                dirnames.remove(dirname)

        # Nom du dossier parent
        parent_folder_name = os.path.basename(dirpath)

        # Suppression des fichiers avec même nom que le dossier parent
        for filename in filenames:
            name_without_ext, ext = os.path.splitext(filename)
            if name_without_ext == parent_folder_name:
                file_to_delete = os.path.join(dirpath, filename)
                print(f"Suppression du fichier : {file_to_delete}")
                os.remove(file_to_delete)