import time
from functools import partial
from typing import TYPE_CHECKING

from PySide6 import QtWidgets, QtGui, QtCore

from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets import widget_components
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets import ui_workers
if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QPixmap)
def add_media_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type, media_id):
    add_media_thumbnail_button(main_window, widget_components.TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type, media_id=media_id)

# Functions to add Buttons with thumbnail for selecting videos/images and faces
@QtCore.Slot(str, QtGui.QPixmap, str, int, int)
def add_webcam_thumbnail_to_target_videos_list(main_window: 'MainWindow', media_path, pixmap, file_type, media_id, webcam_index, webcam_backend):
    add_media_thumbnail_button(main_window, widget_components.TargetMediaCardButton, main_window.targetVideosList, main_window.target_videos, pixmap, media_path=media_path, file_type=file_type, media_id=media_id, is_webcam=True, webcam_index=webcam_index, webcam_backend=webcam_backend)

@QtCore.Slot()
def add_media_thumbnail_to_target_faces_list(main_window: 'MainWindow', cropped_face, embedding_store, pixmap, face_id):
    add_media_thumbnail_button(main_window, widget_components.TargetFaceCardButton, main_window.targetFacesList, main_window.target_faces, pixmap, cropped_face=cropped_face, embedding_store=embedding_store, face_id=face_id )

@QtCore.Slot()
def add_media_thumbnail_to_source_faces_list(main_window: 'MainWindow', media_path, cropped_face, embedding_store, pixmap, face_id):
    add_media_thumbnail_button(main_window, widget_components.InputFaceCardButton, main_window.inputFacesList, main_window.input_faces, pixmap, media_path=media_path, cropped_face=cropped_face, embedding_store=embedding_store, face_id=face_id )

def add_media_thumbnail_button(main_window: 'MainWindow', buttonClass: 'widget_components.CardButton', listWidget: QtWidgets.QListWidget, buttons_list: dict, pixmap, **kwargs):
    print(f"==[DEBUG ADD MINIATURA]==")
    print(f"Classe do botão: {buttonClass.__name__}")
    print(f"media_path: {kwargs.get('media_path')}")
    print(f"file_type: {kwargs.get('file_type')}")
    print(f"media_id: {kwargs.get('media_id')}")
    print(f"pixmap é None? {'SIM' if pixmap is None else 'NÃO'}")
    print(f"pixmap é QPixmap válido? {'SIM' if isinstance(pixmap, QtGui.QPixmap) and not pixmap.isNull() else 'NÃO'}")
    print(f"kwargs: {kwargs}")

    # Define argumentos corretos para cada tipo de botão
    if buttonClass == widget_components.TargetMediaCardButton:
        constructor_args = (
            kwargs.get('media_path'),
            kwargs.get('file_type'),
            kwargs.get('media_id'),
            kwargs.get('is_webcam', False),
            kwargs.get('webcam_index', -1),
            kwargs.get('webcam_backend', -1)
        )
        button_size = QtCore.QSize(90, 90)

    elif buttonClass in (widget_components.TargetFaceCardButton, widget_components.InputFaceCardButton):
        constructor_args = (
            kwargs.get('media_path', ''),
            kwargs.get('cropped_face'),
            kwargs.get('embedding_store'),
            kwargs.get('face_id')
        )
        button_size = QtCore.QSize(70, 70)

    elif buttonClass == widget_components.EmbeddingCardButton:
        constructor_args = (
            kwargs.get('embedding_name'),
            kwargs.get('embedding_store'),
            kwargs.get('embedding_id')
        )
        button_size = QtCore.QSize(70, 70)

    else:
        print(f"[ERRO] Classe de botão não reconhecida: {buttonClass.__name__}")
        return

    # Criação do botão
    button = buttonClass(*constructor_args, main_window=main_window)
    button.set_thumbnail(pixmap, button_size)
    button.setFixedSize(button_size)  # Agora passa o tamanho certo!
    print(f"Botão criado: {button}, type: {type(button)}")
    print(f"pixmap: {pixmap}, pixmap é QPixmap válido? {'SIM' if isinstance(pixmap, QtGui.QPixmap) and not pixmap.isNull() else 'NÃO'}")
    print(f"Tem set_thumbnail? {hasattr(button, 'set_thumbnail')}, Tem setIcon? {hasattr(button, 'setIcon')}")
    button.set_thumbnail(pixmap)
    print(f"Depois do set_thumbnail: iconSize={button.iconSize()}, isCheckable={button.isCheckable()}")
    print(f"==DEBUG add_media_thumbnail_button== kwargs: {kwargs}")

    button.setFixedSize(button_size)
    button.setCheckable(True)

    # Salva o botão na lista de controle
    if buttonClass in [widget_components.TargetFaceCardButton, widget_components.InputFaceCardButton]:
        buttons_list[button.face_id] = button
    elif buttonClass == widget_components.TargetMediaCardButton:
        buttons_list[button.media_id] = button
    elif buttonClass == widget_components.EmbeddingCardButton:
        buttons_list[button.embedding_id] = button

    # Cria o item da lista
    list_item = QtWidgets.QListWidgetItem()
    list_item.setSizeHint(button.sizeHint())
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    listWidget.addItem(list_item)
    listWidget.setItemWidget(list_item, button)

    # Prints para debug depois que tudo foi setado:
    print("TAMANHO FINAL do botão:", button.size())
    print("TAMANHO DO ÍCONE:", button.iconSize())
    print("TAMANHO DA LISTA:", listWidget.size())

    # Essas configs precisam ser sempre aplicadas!
    listWidget.setViewMode(QtWidgets.QListView.IconMode)          # MOSTRAR COMO ÍCONE, NÃO LISTA
    listWidget.setIconSize(button_size)                            # Ícone/thumbnail do tamanho do botão
    listWidget.setGridSize(button_size + QtCore.QSize(10, 10))     # Dá um espaço extra no grid
    listWidget.setWrapping(True)
    listWidget.setFlow(QtWidgets.QListView.LeftToRight)
    listWidget.setResizeMode(QtWidgets.QListView.Adjust)


def create_and_add_embed_button_to_list(main_window: 'MainWindow', embedding_name, embedding_store, embedding_id):
    inputEmbeddingsList = main_window.inputEmbeddingsList
    # Passa l'intero embedding_store
    embed_button = widget_components.EmbeddingCardButton(main_window=main_window, embedding_name=embedding_name, embedding_store=embedding_store, embedding_id=embedding_id)

    button_size = QtCore.QSize(105, 35)  # Adjusted width to fit 3 per row with proper spacing
    embed_button.setFixedSize(button_size)
    
    list_item = QtWidgets.QListWidgetItem(inputEmbeddingsList)
    list_item.setSizeHint(button_size)
    embed_button.list_item = list_item
    list_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    
    inputEmbeddingsList.setItemWidget(list_item, embed_button)
    
    # Configure grid layout for 3x3 minimum grid
    grid_size_with_padding = button_size + QtCore.QSize(4, 4)  # Add padding around buttons
    inputEmbeddingsList.setGridSize(grid_size_with_padding)
    inputEmbeddingsList.setWrapping(True)
    inputEmbeddingsList.setFlow(QtWidgets.QListView.TopToBottom)
    inputEmbeddingsList.setResizeMode(QtWidgets.QListView.Fixed)
    inputEmbeddingsList.setSpacing(2)
    inputEmbeddingsList.setUniformItemSizes(True)
    inputEmbeddingsList.setViewMode(QtWidgets.QListView.IconMode)
    inputEmbeddingsList.setMovement(QtWidgets.QListView.Static)
    
    # Set viewport mode and item size
    viewport_height = 180  # Fixed height for 3 rows (35px + padding per row)
    inputEmbeddingsList.setFixedHeight(viewport_height)
    
    # Calculate grid dimensions
    row_height = viewport_height // 3
    col_width = grid_size_with_padding.width()
    
    # Set minimum width for 3 columns and adjust spacing
    min_width = (3 * col_width) + 16  # Add extra padding for better spacing between columns
    inputEmbeddingsList.setMinimumWidth(min_width)
    
    # Configure scrolling behavior
    inputEmbeddingsList.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    inputEmbeddingsList.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    inputEmbeddingsList.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
    inputEmbeddingsList.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
    
    # Set layout direction to ensure proper filling
    inputEmbeddingsList.setLayoutDirection(QtCore.Qt.LeftToRight)
    inputEmbeddingsList.setLayoutMode(QtWidgets.QListView.Batched)

    main_window.merged_embeddings[embed_button.embedding_id] = embed_button

def clear_stop_loading_target_media(main_window: 'MainWindow'):
    if main_window.video_loader_worker:
        main_window.video_loader_worker.stop()
        main_window.video_loader_worker.terminate()
        main_window.video_loader_worker = False
        time.sleep(0.5)
        main_window.targetVideosList.clear()

@QtCore.Slot()
def select_target_medias(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=None):
    files_list = files_list or []
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(dir=main_window.last_target_media_folder_path)
        if not folder_name:
            return
        main_window.labelTargetVideosPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelTargetVideosPath.setToolTip(folder_name)
        main_window.last_target_media_folder_path = folder_name

    elif source_type=='files':
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        # Get Folder name from the first file
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.labelTargetVideosPath.setText(file_dir) #Just a temp text until i think of something better
        main_window.labelTargetVideosPath.setToolTip(file_dir)
        main_window.last_target_media_folder_path = file_dir

    clear_stop_loading_target_media(main_window)
    card_actions.clear_target_faces(main_window)
    
    main_window.selected_video_button = False
    main_window.target_videos = {}

    main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=main_window, folder_name=folder_name, files_list=files_list)
    main_window.video_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_target_videos_list, main_window))
    main_window.video_loader_worker.start()

@QtCore.Slot()
def load_target_webcams(main_window: 'MainWindow',):
    if main_window.filterWebcamsCheckBox.isChecked():
        main_window.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=main_window, webcam_mode=True)
        main_window.video_loader_worker.webcam_thumbnail_ready.connect(partial(add_webcam_thumbnail_to_target_videos_list, main_window))
        main_window.video_loader_worker.start()
    else:
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, True)
        for _, target_video in main_window.target_videos.copy().items(): #Use a copy of the dict to prevent Dictionary changed during iteration exceptions
            if target_video.file_type == 'webcam':
                target_video.remove_target_media_from_list()
                if target_video == main_window.selected_video_button:
                    main_window.selected_video_button = False
        main_window.placeholder_update_signal.emit(main_window.targetVideosList, False)

def clear_stop_loading_input_media(main_window: 'MainWindow'):
    if main_window.input_faces_loader_worker:
        main_window.input_faces_loader_worker.stop()
        main_window.input_faces_loader_worker.terminate()
        main_window.input_faces_loader_worker = False
        time.sleep(0.5)
        main_window.inputFacesList.clear()

@QtCore.Slot()
def select_input_face_images(main_window: 'MainWindow', source_type='folder', folder_name=False, files_list=None):
    files_list = files_list or []
    if source_type=='folder':
        folder_name = QtWidgets.QFileDialog.getExistingDirectory(dir=main_window.last_input_media_folder_path)
        if not folder_name:
            return
        main_window.labelInputFacesPath.setText(misc_helpers.truncate_text(folder_name))
        main_window.labelInputFacesPath.setToolTip(folder_name)
        main_window.last_input_media_folder_path = folder_name

    elif source_type=='files':
        files_list = QtWidgets.QFileDialog.getOpenFileNames()[0]
        if not files_list:
            return
        file_dir = misc_helpers.get_dir_of_file(files_list[0])
        main_window.labelInputFacesPath.setText(file_dir) #Just a temp text until i think of something better
        main_window.labelInputFacesPath.setToolTip(file_dir)
        main_window.last_input_media_folder_path = file_dir

    clear_stop_loading_input_media(main_window)
    card_actions.clear_input_faces(main_window)
    main_window.input_faces_loader_worker = ui_workers.InputFacesLoaderWorker(main_window=main_window, folder_name=folder_name, files_list=files_list)
    main_window.input_faces_loader_worker.thumbnail_ready.connect(partial(add_media_thumbnail_to_source_faces_list, main_window))
    main_window.input_faces_loader_worker.start()

def set_up_list_widget_placeholder(main_window, list_widget):
    drop_text = main_window.tr("Drop Files")
    or_text = main_window.tr("or")
    click_text = main_window.tr("Click here to Select a Folder")

    placeholder_label = getattr(list_widget, 'placeholder_label', None)
    if placeholder_label is None:
        placeholder_label = QtWidgets.QLabel(list_widget)
        placeholder_label.setStyleSheet("color: gray; font-size: 15px; font-weight: bold;")
        placeholder_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QtWidgets.QVBoxLayout(list_widget)
        layout.addWidget(placeholder_label)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        list_widget.placeholder_label = placeholder_label
        list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    placeholder_label.setText(
        f"<html><body style='text-align:center;'>"
        f"<p>{drop_text}</p>"
        f"<p><b>{or_text}</b></p>"
        f"<p>{click_text}</p>"
        f"</body></html>"
    )
    # Vai esconder quando tiver pelo menos 1 item na lista
    placeholder_label.setVisible(list_widget.count() == 0)


def select_output_media_folder(main_window: 'MainWindow'):
    folder_name = QtWidgets.QFileDialog.getExistingDirectory()
    if folder_name:
        main_window.outputFolderLineEdit.setText(folder_name)
        common_widget_actions.create_control(main_window, 'OutputMediaFolder', folder_name)