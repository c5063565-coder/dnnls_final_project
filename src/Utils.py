def save_checkpoint(model, optimizer, epoch, loss, filename):
    path = os.path.join(FOLDERS['checkpoints'], filename)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict() if optimizer else None,
        'loss':loss,
    }, path)
    print(f"Checkpoint saved: checkpoints/{filename}")
 
 
def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    path = os.path.join(FOLDERS['checkpoints'], filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and ckpt.get('optimizer_state_dict'):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"Loaded: {filename}  (epoch {ckpt.get('epoch', 0)})")
    return model, optimizer, ckpt.get('epoch', 0), ckpt.get('loss', None)
 
 
def save_log(lines, experiment):
    filename = f"training_log_{experiment}.txt"
    path= os.path.join(FOLDERS[experiment], filename)
    content= '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(content)
    with open(filename, 'w') as f:
        f.write(content)
 
    print(f"Log saved → experiments/{experiment}/{filename}")
 
 
def save_figure(fig, filename, experiment):
    path = os.path.join(FOLDERS[experiment], filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Figure saved → experiments/{experiment}/{filename}")
 
 
def parse_gdi_text(text):
    soup = BeautifulSoup(text, 'html.parser')
    images = []
 
    for gdi in soup.find_all('gdi'):
        image_id = None
        if gdi.attrs:
            for attr_name in gdi.attrs:
                if 'image' in attr_name.lower():
                    image_id = attr_name.replace('image', '')
                    break
        if not image_id:
            match = re.search(r'<gdi\s+image(\d+)', str(gdi))
            if match:
                image_id = match.group(1)
        if not image_id:
            image_id = str(len(images) + 1)
 
        images.append({
            'image_id':image_id,
            'description': gdi.get_text().strip(),
            'objects':[o.get_text().strip() for o in gdi.find_all('gdo')],
            'actions':[a.get_text().strip() for a in gdi.find_all('gda')],
            'locations':[l.get_text().strip() for l in gdi.find_all('gdl')],
        })
 
    return images
 
 
def show_image(ax, image):
    ax.imshow(image.permute(1, 2, 0).cpu().numpy().clip(0, 1))
 
 
def _parse_markdown_table(block):
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
 
    if len(table_lines) < 3:
        return []
 
    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows= []
 
    for line in table_lines[2:]:
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) == len(headers):
            rows.append(dict(zip(headers, cols)))
 
    return rows
 
 
def parse_cot_grounding(chain_of_thought):
    frames = {}
    pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(pattern.finditer(chain_of_thought or ""))
 
    for i, m in enumerate(matches):
        idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(chain_of_thought)
        section = chain_of_thought[start:end]
 
        frames[idx] = {"characters": [], "objects": []}
 
        for tag, key in [("Characters", "characters"), ("Objects", "objects")]:
            hit = re.search(
                rf"###\s*{tag}(.*?)(?=\n###|\n##|$)",
                section, re.DOTALL
            )
            if hit:
                id_key = "Character ID" if key == "characters" else "Object ID"
                for row in _parse_markdown_table(hit.group(1)):
                    eid = row.get(id_key, "").strip()
                    bbox = row.get("Bounding Box", "").strip()
                    if eid and bbox:
                        try:
                            x1, y1, x2, y2 = [int(v) for v in bbox.split(",")]
                            frames[idx][key].append({
                                "id":   eid,
                                "bbox": [x1, y1, x2, y2]
                            })
                        except:
                            pass
 
    return frames
 
 
def crop_and_resize(pil_img, bbox, out_hw=IMAGE_HW):
    W, H = pil_img.size
    x1, y1, x2, y2 = bbox
 
    x1, x2 = max(0, x1), min(W-1, x2)
    y1, y2 = max(0, y1), min(H-1, y2)
 
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
 
    crop = pil_img.crop((x1, y1, x2, y2))
    return transforms.Compose([
        transforms.Resize(out_hw),
        transforms.ToTensor()
    ])(crop)
 
 
def pick_reid_pair(frames_cot):
    import random
    id_to_dets = {}
 
    for f_idx, content in frames_cot.items():
        for det in content.get("characters", []) + content.get("objects", []):
            eid  = det.get("id")
            bbox = det.get("bbox")
            if eid and bbox:
                id_to_dets.setdefault(eid, []).append((f_idx, bbox))
 
    candidates = [e for e, d in id_to_dets.items() if len(d) >= 2]
 
    if not candidates:
        return None
 
    eid = random.choice(candidates)
    (f1, b1), (f2, b2) = random.sample(id_to_dets[eid], 2)
    return f1, f2, b1, b2, eid
 
 
def extract_cot_text_for_frame(cot, frame_idx, max_chars=600):
    if not cot:
        return ""
    pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(pattern.finditer(cot))
 
    for i, m in enumerate(matches):
        if int(m.group(1)) - 1 == frame_idx:
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(cot)
            target = cot[start:end]
 
            lines = [
                l for l in target.splitlines()
                if not l.strip().startswith("|")
                and set(l.strip()) > set("-|:")
            ]
            text = " ".join(l.strip() for l in lines if l.strip())
            return re.sub(r"\s+", " ", text).strip()[:max_chars]
 
    return ""
 
 
def tensor_to_np(t):
    return t.permute(1, 2, 0).cpu().numpy().clip(0, 1)
 
class SequencePredictionDataset(Dataset):
    def __init__(self, dataset, tokenizer, K=4, max_len=MAX_LEN, image_hw=IMAGE_HW):
        self.dataset = dataset
        self.tokenizer= tokenizer
        self.K = K
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize(image_hw),
            transforms.ToTensor(),
        ])
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        story = self.dataset[idx]
        frames = story["images"]
        image_attrs = parse_gdi_text(story["story"])
        cot= story.get("chain_of_thought", "")
        cot_frames = parse_cot_grounding(cot)
        frame_tensors = []
        description_list = []
 
        for fi in range(self.K):
            img = self.transform(FT.equalize(frames[fi]))
            frame_tensors.append(img)
            desc = image_attrs[fi]["description"]
            if USE_COT_TEXT:
                cot_txt = extract_cot_text_for_frame(cot, fi)
                if cot_txt:
                    desc = desc + " [COT] " + cot_txt
            ids = self.tokenizer(
                desc,
                return_tensors = "pt",
                padding = "max_length",
                truncation = True,
                max_length = self.max_len
            ).input_ids.squeeze(0)
            description_list.append(ids)
        image_target = self.transform(FT.equalize(frames[self.K]))
        target_desc  = image_attrs[self.K]["description"]
        target_ids   = self.tokenizer(
            target_desc,
            return_tensors = "pt",
            padding = "max_length",
            truncation = True,
            max_length = self.max_len
        ).input_ids
        roi1 = torch.zeros(3, *IMAGE_HW)
        roi2 = torch.zeros(3, *IMAGE_HW)
        roi_valid = 0
        roi_frame_idx = -1
        ent_id = ""
 
        pair = pick_reid_pair(cot_frames)
        if pair:
            f1, f2, b1, b2, eid = pair
            if f1 < self.K and f2 < self.K:
                try:
                    roi1 = crop_and_resize(frames[f1], b1, IMAGE_HW)
                    roi2 = crop_and_resize(frames[f2], b2, IMAGE_HW)
                    roi_valid = 1
                    roi_frame_idx = f1
                    ent_id = eid
                except:
                    pass
        return (
            torch.stack(frame_tensors),
            torch.stack(description_list),
            image_target,
            target_ids,
            roi1,
            roi2,
            torch.tensor(roi_valid),
            torch.tensor(roi_frame_idx),
            ent_id
        )
 
 
class TextTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        attrs = parse_gdi_text(self.dataset[idx]["story"])
        fi= np.random.randint(0, 5)  # pick a random frame
        return attrs[fi]["description"] # return just the text string
 
 
class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset= dataset
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_HW),
            transforms.ToTensor(),
        ])
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        frames = self.dataset[idx]["images"]
        fi = torch.randint(0, 5, (1,)).item()   
        return (self.transform(frames[fi]),)          
