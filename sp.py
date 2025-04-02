import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch

# ----------------------------
# Configurações e Parâmetros
# ----------------------------
VIDEO_IN = "CFBB vs UNION TARBES LOURDES PYRENEES BASKET Men's Pro Basketball - Tactical.mp4"
VIDEO_OUT = "pitch_supt15.mp4"  # Exemplo, ajusta conforme o tamanho
PITCH_FILE = 'pitch.npy'
MIN_MATCH_COUNT = 10
FRAME_INDICES = [104700, 104775, 104810]  # Frames de referência
SIZE_RATIO = 15  # Fator de redimensionamento (ajustado depois para divisão)
CONF_THRESH = 10 / 100.0  # Convertendo para proporção

# ----------------------------
# Funções Auxiliares
# ----------------------------
def load_reference_images_and_annotations(frame_indices, img_prefix="img_", annot_prefix="annots_"):
    """
    Carrega as imagens de referência e as respectivas anotações.
    
    Args:
        frame_indices (list): Lista de frames de referência.
        img_prefix (str): Prefixo dos nomes dos arquivos de imagem.
        annot_prefix (str): Prefixo dos nomes dos arquivos de anotações.
    
    Returns:
        tuple: (imagens, anotações, índices das anotações)
    """
    images = []
    annotations = []
    annotations_idx = []
    for i in frame_indices:
        img_path = f"{img_prefix}{i}.png"
        annot_path = f"{annot_prefix}{i}.npy"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")
        images.append(img)
        annot = np.load(annot_path)
        annotations.append(annot[:, 1:])  # Assume que a primeira coluna é um flag
        annotations_idx.append(np.where(annot[:, 0])[0])
    return images, annotations, annotations_idx

def calculate_histograms(images):
    """
    Calcula e empilha os histogramas de cada imagem.
    
    Args:
        images (list): Lista de imagens.
        
    Returns:
        np.array: Histograma empilhado de todas as imagens.
    """
    return np.hstack([cv2.calcHist([img], [0], None, [256], [0, 256]) for img in images])

def resize_and_convert(images, size_ratio):
    """
    Redimensiona as imagens e converte de BGR para RGB.
    
    Args:
        images (list): Lista de imagens.
        size_ratio (float): Fator de redimensionamento.
        
    Returns:
        list: Lista de imagens redimensionadas e convertidas para RGB.
    """
    resized_rgb = []
    h, w = images[0].shape[:2]
    if size_ratio != 1:
        new_size = (int(w / size_ratio), int(h / size_ratio))
        for img in images:
            img_resized = cv2.resize(img, new_size)
            resized_rgb.append(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    else:
        resized_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    return resized_rgb

def detect_keypoints_and_descriptors(processor, model, rgbs, device, conf_thresh):
    """
    Detecta keypoints e extrai descritores das imagens usando o SuperPoint.
    
    Args:
        processor: Processador de imagem.
        model: Modelo SuperPoint.
        rgbs (list): Lista de imagens RGB.
        device (str): Dispositivo (CPU ou GPU).
        conf_thresh (float): Limite de confiança para seleção de keypoints.
        
    Returns:
        tuple: Listas de keypoints e descritores para cada imagem.
    """
    with torch.no_grad():
        inputs = processor(rgbs, return_tensors="pt").to(device)
        outputs = model(**inputs)
        # Usando tamanho de imagem fixo (ajuste se necessário)
        image_sizes = torch.tile(torch.tensor([1, 1]), (len(rgbs), 1)).to(device)
        outputs = processor.post_process_keypoint_detection(outputs, image_sizes)
    
    keypoints_list = []
    descriptors_list = []
    for output in outputs:
        kp = output['keypoints'].cpu().numpy()
        desc = output['descriptors'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        # Filtrar keypoints com base na confiança
        valid = scores > conf_thresh
        kp = kp[valid]
        desc = desc[valid]
        # Filtrar pontos fora da área de interesse (ajuste conforme necessário)
        in_bounds = np.logical_not((kp[:, 1] > 875) & (kp[:, 0] < 325))
        keypoints_list.append(kp[in_bounds])
        descriptors_list.append(desc[in_bounds])
    return keypoints_list, descriptors_list

def best_match(new_img, ref_hist):
    """
    Encontra o melhor match da imagem com base no histograma.
    
    Args:
        new_img: Imagem nova.
        ref_hist: Histograma de referência.
    
    Returns:
        tuple: Índice da melhor correspondência e array com as similaridades.
    """
    new_hist = cv2.calcHist([new_img], [0], None, [256], [0, 256])
    match_probs = [cv2.matchTemplate(hist, new_hist, cv2.TM_CCOEFF_NORMED)[0][0] for hist in ref_hist.T]
    match_probs = np.array(match_probs)
    return np.argmax(match_probs), match_probs

def match_keypoints(descriptors_ref, descriptors, keypoints_ref, keypoints, index_params, search_params, min_match_count):
    """
    Realiza o matching dos descritores e calcula a homografia se houver matches suficientes.
    
    Args:
        descriptors_ref: Descritores da imagem de referência.
        descriptors: Descritores da imagem atual.
        keypoints_ref: Keypoints da imagem de referência.
        keypoints: Keypoints da imagem atual.
        index_params: Parâmetros para o FLANN.
        search_params: Parâmetros para o FLANN.
        min_match_count: Número mínimo de matches para considerar válido.
        
    Returns:
        tuple: Homografia, lista de bons matches e a máscara de inliers.
    """
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_ref, descriptors, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) > min_match_count:
        src_pts = np.float32([keypoints_ref[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return homography, good_matches, mask
    else:
        return None, good_matches, None

def process_video_frames(video_in, init_frame, num_frames, processor, model, ref_hist,
                         annotations, annotations_idx, keypoints_ref_list, descriptors_ref_list,
                         size_ratio, conf_thresh, device, index_params, search_params):
    """
    Processa os frames do vídeo, detectando keypoints, realizando o matching e calculando as transformações.
    
    Args:
        video_in (str): Caminho do vídeo de entrada.
        init_frame (int): Frame inicial para o processamento.
        num_frames (int): Número de frames a processar.
        (outros parâmetros): São utilizados conforme as funções definidas.
        
    Returns:
        list: Lista de matrizes de homografia calculadas para cada frame.
    """
    video_capture = cv2.VideoCapture(video_in)
    if not video_capture.isOpened():
        raise Exception("Não foi possível abrir o vídeo.")
    
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    Hs = []
    t_match_total = 0
    batch_size = 4

    for i in range(0, num_frames, batch_size):
        frames = []
        hist_matches = []
        # Processa em lote (batch)
        for j in range(batch_size):
            ret, frame = video_capture.read()
            if not ret:
                break
            i_match, _ = best_match(frame, ref_hist)
            hist_matches.append(i_match)
            frames.append(frame)
        if not frames:
            break

        rgbs = resize_and_convert(frames, size_ratio if size_ratio == 1 else size_ratio / 10)
        keypoints_list, descriptors_list = detect_keypoints_and_descriptors(processor, model, rgbs, device, conf_thresh)

        for i_match, kp, desc in zip(hist_matches, keypoints_list, descriptors_list):
            t_start = time.time()
            homography, good_matches, mask = match_keypoints(
                descriptors_ref_list[i_match], desc,
                keypoints_ref_list[i_match], kp,
                index_params, search_params, MIN_MATCH_COUNT)
            t_match_total += time.time() - t_start
            if homography is not None:
                # Ajusta a transformação considerando o redimensionamento
                M = np.diag([size_ratio, size_ratio, 1]) @ homography @ np.diag([1/size_ratio, 1/size_ratio, 1])
                # Aplica a homografia nas anotações
                new_pts = cv2.perspectiveTransform(annotations[i_match].reshape(-1, 1, 2), M).squeeze()
                # Se necessário, aqui você pode escolher usar os pontos de pitch.npy ou os das anotações
                pitch_in = annotations[i_match]  # ou substitua por: np.load(PITCH_FILE)[annotations_idx[i_match]]
                new_in = new_pts[annotations_idx[i_match]]
                M2img, _ = cv2.findHomography(pitch_in, new_in, cv2.RANSAC)
                M2pitch, _ = cv2.findHomography(new_in, pitch_in, cv2.RANSAC)
                Hs.append(np.stack((M, M2img, M2pitch)))
            else:
                Hs.append(None)
        print(f"Processados {i+batch_size} frames, tempo de matching acumulado: {t_match_total:.2f} s")
    
    video_capture.release()
    return Hs

def write_video(video_in, output_video, init_frame, num_frames, Hs, annotations, annotations_idx):
    """
    Aplica as transformações calculadas aos frames e gera o vídeo final.
    
    Args:
        video_in (str): Caminho do vídeo original.
        output_video (str): Caminho do vídeo de saída.
        init_frame (int): Frame inicial.
        num_frames (int): Número de frames a processar.
        Hs (list): Lista das homografias calculadas.
        annotations (list): Lista de anotações para os frames de referência.
        annotations_idx (list): Índices das anotações relevantes.
    """
    video_capture = cv2.VideoCapture(video_in)
    if not video_capture.isOpened():
        raise Exception("Não foi possível abrir o vídeo para escrita.")
    
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter("results.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (0, 0, 255)
    thickness = 3
    lineType = 2

    for i in range(num_frames):
        ret, frame = video_capture.read()
        if not ret:
            break
        
        if Hs[i] is not None:
            M = Hs[i][0]
            i_match = int(M[2, 2])
            new_pts = cv2.perspectiveTransform(annotations[i_match].reshape(-1, 1, 2), M).squeeze()
            for pt in new_pts.astype(np.int16):
                cv2.circle(frame, tuple(pt), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"{i}", (100, 100), font, fontScale, fontColor, thickness, lineType)
        video_writer.write(frame)
    
    video_capture.release()
    video_writer.release()
    
    # Converter para o formato final usando ffmpeg
    cmd = f"ffmpeg -v quiet -i results.avi -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {output_video}"
    os.system(cmd)
    os.remove("results.avi")
    print("Vídeo convertido com sucesso!")

# ----------------------------
# Execução Principal
# ----------------------------
def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    model = model.to(device)
    
    # Carrega imagens e anotações de referência
    ref_imgs, annotations, annotations_idx = load_reference_images_and_annotations(FRAME_INDICES)
    ref_hist = calculate_histograms(ref_imgs)
    # Ajusta o tamanho conforme SIZE_RATIO (dividindo por 10 se necessário)
    rgbs_ref = resize_and_convert(ref_imgs, SIZE_RATIO if SIZE_RATIO == 1 else SIZE_RATIO / 10)
    keypoints_ref_list, descriptors_ref_list = detect_keypoints_and_descriptors(processor, model, rgbs_ref, device, CONF_THRESH)
    
    # Processa frames do vídeo para calcular as homografias
    init_frame = 100_000
    num_frames = 2000
    index_params = dict(algorithm=1, trees=5)  # FLANN com KDTree
    search_params = dict(checks=50)
    
    Hs = process_video_frames(VIDEO_IN, init_frame, num_frames, processor, model, ref_hist,
                              annotations, annotations_idx, keypoints_ref_list, descriptors_ref_list,
                              SIZE_RATIO, CONF_THRESH, device, index_params, search_params)
    
    # Salva as homografias calculadas
    Hs_name = f"Hs_supt{SIZE_RATIO}.npy"
    np.save(Hs_name, Hs)
    
    # Gera e escreve o vídeo final com as transformações aplicadas
    write_video(VIDEO_IN, VIDEO_OUT, init_frame, num_frames, Hs, annotations, annotations_idx)
    
if __name__ == "__main__":
    main()
