def pretrain_text_external():
    print("External Dataset Pretraining (Text)")
    print("=" * 55)
    print(f"Encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}")
    print(f"Experiment: {EXP_NAME}\n")

    for p in text_ae.parameters():
        p.requires_grad = True
    print(f"Parameters unfrozen: {sum(p.numel() for p in text_ae.parameters()):,}\n")

    loss_fn_ext = nn.CrossEntropyLoss(
        ignore_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    EXT_LR  = 0.001 if USE_TRANSFORMER_ENCODER else 0.0003
    opt_ext = torch.optim.Adam(text_ae.parameters(), lr=EXT_LR)

    log_lines = [
        f"Experiment: {EXP_NAME}",
        f"External Dataset Pretraining",
        f"Encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}",
        f"LR: {EXT_LR}",
        "-" * 55,
    ]

    wiki_losses = []

    if USE_TRANSFORMER_ENCODER:
        print("Stage A: WikiText-2 Pretraining (Transformer only)")
        print("─" * 55)
        print("Wikipedia text-teaches general English structure.\n")

        wiki = hf_load("wikitext", "wikitext-2-raw-v1", split="train")
        wiki_texts = [
            row['text'] for row in wiki
            if len(row['text'].strip()) > 30
        ]
        print(f"WikiText-2 sentences: {len(wiki_texts):,}\n")
        N_WIKI_EPOCHS= 3
        SAMPLES_PER_EPOCH= 500
        WIKI_BATCH_SIZE= 4
        sched_wiki = torch.optim.lr_scheduler.StepLR(
            opt_ext, step_size=2, gamma=0.5
        )
        log_lines.append("Stage A-WikiText-2")
        log_lines.append(f"Epochs: {N_WIKI_EPOCHS}")
        log_lines.append(f"Samples/epoch : {SAMPLES_PER_EPOCH}")
        log_lines.append("-" * 55)

        for epoch in range(N_WIKI_EPOCHS):
            text_ae.train()
            epoch_loss= 0.0
            batch_count = 0
            t0 = time.time()

            sample_texts = np.random.choice(
                wiki_texts, size=SAMPLES_PER_EPOCH, replace=False
            )

            for i in range(0, len(sample_texts), WIKI_BATCH_SIZE):
                batch = list(sample_texts[i : i + WIKI_BATCH_SIZE])
                ids= tokenizer(
                    batch,
                    return_tensors = "pt",
                    padding= "max_length",
                    truncation= True,
                    max_length= MAX_LEN
                ).input_ids.to(device)

                opt_ext.zero_grad()
                outputs = text_ae(ids, ids)
                loss= loss_fn_ext(
                    outputs.reshape(-1, tokenizer.vocab_size),
                    ids[:, 1:].reshape(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(text_ae.parameters(), 1.0)
                opt_ext.step()

                epoch_loss+= loss.item()
                batch_count += 1

            sched_wiki.step()
            avg= epoch_loss / batch_count
            elapsed = time.time() - t0
            wiki_losses.append(avg)

            line = (f"Epoch {epoch+1}/{N_WIKI_EPOCHS} | "
                    f"avg_loss={avg:.4f} | "
                    f"last_loss={loss.item():.4f} | "
                    f"time={elapsed:.1f}s")
            print(line)
            log_lines.append(line)

            save_checkpoint(
                text_ae, opt_ext, epoch+1, loss,
                filename=f"text_wiki_epoch{epoch+1}.pth"
            )

        log_lines.append(f"Stage A initial : {wiki_losses[0]:.4f}")
        log_lines.append(f"Stage A final: {wiki_losses[-1]:.4f}")
        log_lines.append("")
        print(f"\nStage A complete.")
        print(f"{wiki_losses[0]:.4f} → {wiki_losses[-1]:.4f}\n")

    print("Stage B: TinyStories Pretraining (ALL encoders)")
    print("─" * 55)
    print("Loading TinyStories")
    tiny= hf_load("roneneldan/TinyStories", split="train")
    tiny_texts = []

    for row in tiny:
        story = row['text'].strip()
        if len(story) > 30:
            sentences = [s.strip() for s in story.split('.') if len(s.strip()) > 20]
            tiny_texts.extend(sentences)
        if len(tiny_texts) >= 50000:
            break
    print(f"TinyStories sentences extracted: {len(tiny_texts):,}\n")

    N_TINY_EPOCHS= 10
    TINY_SAMPLES= 500
    TINY_BATCH_SIZE= 4
    tiny_losses= []

    sched_tiny= torch.optim.lr_scheduler.StepLR(
        opt_ext, step_size=2, gamma=0.5
    )
    log_lines.append("Stage B-TinyStories")
    log_lines.append(f"Epochs: {N_TINY_EPOCHS}")
    log_lines.append(f"Samples/epoch :{TINY_SAMPLES}")
    log_lines.append("-" * 55)

    print(f"Training {N_TINY_EPOCHS} epochs x {TINY_SAMPLES} samples...\n")

    for epoch in range(N_TINY_EPOCHS):
        text_ae.train()
        epoch_loss= 0.0
        batch_count= 0
        t0= time.time()

        sample_texts = np.random.choice(
            tiny_texts, size=TINY_SAMPLES, replace=False
        )

        for i in range(0, len(sample_texts), TINY_BATCH_SIZE):
            batch = list(sample_texts[i : i + TINY_BATCH_SIZE])
            ids= tokenizer(
                batch,
                return_tensors= "pt",
                padding= "max_length",
                truncation= True,
                max_length= MAX_LEN
            ).input_ids.to(device)

            opt_ext.zero_grad()
            outputs = text_ae(ids, ids)
            loss= loss_fn_ext(
                outputs.reshape(-1, tokenizer.vocab_size),
                ids[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_ae.parameters(), 1.0)
            opt_ext.step()

            epoch_loss += loss.item()
            batch_count += 1

        sched_tiny.step()
        avg = epoch_loss / batch_count
        elapsed = time.time() - t0
        tiny_losses.append(avg)

        line = (f"Epoch {epoch+1}/{N_TINY_EPOCHS} | "
                f"avg_loss={avg:.4f} | "
                f"last_loss={loss.item():.4f} | "
                f"time={elapsed:.1f}s")
        print(line)
        log_lines.append(line)

        save_checkpoint(
            text_ae, opt_ext, epoch+1, loss,
            filename = f"text_tiny_epoch{epoch+1}_"
                       f"{'transformer' if USE_TRANSFORMER_ENCODER else 'lstm'}.pth"
        )

    log_lines.append(f"Stage B initial : {tiny_losses[0]:.4f}")
    log_lines.append(f"Stage B final: {tiny_losses[-1]:.4f}")

    print(f"\nStage B complete.")
    print(f"{tiny_losses[0]:.4f} → {tiny_losses[-1]:.4f}")

    if USE_TRANSFORMER_ENCODER:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(
            range(1, N_WIKI_EPOCHS+1), wiki_losses,
            marker='o', linewidth=2, color='steelblue', label='WikiText-2'
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Stage A: WikiText-2\n(Transformer)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(
            range(1, N_TINY_EPOCHS+1), tiny_losses,
            marker='s', linewidth=2, color='seagreen', label='TinyStories'
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Stage B: TinyStories\n(Transformer)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"External Text Pretraining — {EXP_NAME}", fontsize=13)

    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            range(1, N_TINY_EPOCHS+1), tiny_losses,
            marker='s', linewidth=2, color='seagreen', label='TinyStories'
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"TinyStories Pretraining — {EXP_NAME}\n(LSTM)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "cell12_external_pretraining_loss.png", experiment=EXP_FOLDER)
    plt.show()
    enc_tag = 'transformer' if USE_TRANSFORMER_ENCODER else 'lstm'
    save_checkpoint(
        text_ae, opt_ext, N_TINY_EPOCHS, loss,
        filename = f"text_external_final_{enc_tag}.pth"
    )

    log_lines.append("")
    log_lines.append("-" * 55)
    if USE_TRANSFORMER_ENCODER:
        log_lines.append(f"WikiText-2: {wiki_losses[0]:.4f} → {wiki_losses[-1]:.4f}")
    log_lines.append(f"TinyStories: {tiny_losses[0]:.4f} → {tiny_losses[-1]:.4f}")
    log_lines.append("StoryReasoning fine-tuning.")
    save_log(log_lines, experiment=EXP_FOLDER)

    if USE_TRANSFORMER_ENCODER:
        print(f"WikiText-2  : {wiki_losses[0]:.4f} → {wiki_losses[-1]:.4f}")
    print(f"TinyStories : {tiny_losses[0]:.4f} → {tiny_losses[-1]:.4f}")
def finetune_text_story():
    print("StoryReasoning Text Fine-tuning")
    print("=" * 55)
    print(f"Encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}")
    print(f"Experiment: {EXP_NAME}")
    print()
    print("Transfer learning final stage:")
    print("Adapts text encoder to exact story description domain.")

    for p in text_ae.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in text_ae.parameters() if p.requires_grad)
    print(f"Trainable parameters : {trainable:,}\n")

    STORY_LR= 0.0003 if USE_TRANSFORMER_ENCODER else 0.0001

    loss_fn_story = nn.CrossEntropyLoss(
        ignore_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    opt_story = torch.optim.Adam(text_ae.parameters(), lr=STORY_LR)
    sched_story = torch.optim.lr_scheduler.StepLR(
        opt_story, step_size=2, gamma=0.5
    )

    N_STORY_EPOCHS = 10

    try:
        log_path = os.path.join(
            FOLDERS[EXP_FOLDER],
            f"training_log_{EXP_FOLDER}.txt"
        )
        with open(log_path, 'r') as f:
            existing = f.read().splitlines()
        log_lines = existing
        log_lines.append("")
        print("Loaded existing log appending.")
    except:
        log_lines = [
            f"Experiment : {EXP_NAME}",
            f"Encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}",
            "-" * 55,
        ]
        print("Starting fresh log.")

    text_dataset = TextTaskDataset(train_dataset)
    val_size_text = int(0.1 * len(text_dataset))
    train_size_txt = len(text_dataset) - val_size_text

    txt_train_sub, txt_val_sub = random_split(
        text_dataset,
        [train_size_txt, val_size_text],
        generator = torch.Generator().manual_seed(42)
    )

    txt_train_loader = DataLoader(
        txt_train_sub, batch_size=4, shuffle=True)
    txt_val_loader = DataLoader(
        txt_val_sub,   batch_size=4, shuffle=False)

    log_lines.append("StoryReasoning Fine-tuning")
    log_lines.append("=" * 55)
    log_lines.append(f"Encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}")
    log_lines.append(f"Epochs: {N_STORY_EPOCHS}")
    log_lines.append(f"LR: {STORY_LR}")
    log_lines.append(f"Batch size: {txt_train_loader.batch_size}")
    log_lines.append("-" * 55)

    story_losses = []
    story_val_losses = []

    print(f"Text train samples: {len(txt_train_sub):,}")
    print(f"Text val samples: {len(txt_val_sub):,}")
    print(f"\nFine-tuning for {N_STORY_EPOCHS} epochs...\n")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(N_STORY_EPOCHS):
        text_ae.train()
        train_loss = 0.0
        batch_count = 0
        t0 = time.time()

        for desc in txt_train_loader:
            ids = tokenizer(
                desc,
                return_tensors = "pt",
                padding = "max_length",
                truncation = True,
                max_length = MAX_LEN
            ).input_ids.to(device)

            opt_story.zero_grad()
            outputs = text_ae(ids, ids)
            loss = loss_fn_story(
                outputs.reshape(-1, tokenizer.vocab_size),
                ids[:, 1:].reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_ae.parameters(), 1.0)
            opt_story.step()
            train_loss += loss.item()
            batch_count += 1

        avg_train = train_loss / batch_count

        text_ae.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for desc in txt_val_loader:
                ids = tokenizer(
                    desc,
                    return_tensors = "pt",
                    padding = "max_length",
                    truncation = True,
                    max_length = MAX_LEN
                ).input_ids.to(device)

                outputs = text_ae(ids, ids)
                v_loss = loss_fn_story(
                    outputs.reshape(-1, tokenizer.vocab_size),
                    ids[:, 1:].reshape(-1)
                )
                val_loss += v_loss.item()
                val_batches += 1

        avg_val = val_loss / val_batches
        elapsed = time.time() - t0

        story_losses.append(avg_train)
        story_val_losses.append(avg_val)

        sched_story.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            save_checkpoint(
                text_ae, opt_story, epoch+1, loss,
                filename = f"text_story_best_"
                           f"{'transformer' if USE_TRANSFORMER_ENCODER else 'lstm'}.pth"
            )

        line = (f"Epoch {epoch+1}/{N_STORY_EPOCHS} | "
                f"train={avg_train:.4f} | "
                f"val={avg_val:.4f} | "
                f"time={elapsed:.1f}s"
                f"{'  ← best' if epoch+1 == best_epoch else ''}")
        print(line)
        log_lines.append(line)

        save_checkpoint(
            text_ae, opt_story, epoch+1, loss,
            filename = f"text_story_epoch{epoch+1}_"
                       f"{'transformer' if USE_TRANSFORMER_ENCODER else 'lstm'}.pth"
        )

    print(f"\nLoading best checkpoint (epoch {best_epoch}, val={best_val_loss:.4f})")
    load_checkpoint(
        text_ae,
        filename = f"text_story_best_"
                   f"{'transformer' if USE_TRANSFORMER_ENCODER else 'lstm'}.pth"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, N_STORY_EPOCHS+1)

    ax.plot(epochs, story_losses,
            marker='o', linewidth=2,
            color='steelblue', label='Train Loss')
    ax.plot(epochs, story_val_losses,
            marker='s', linewidth=2,
            color='darkorange', label='Val Loss')
    ax.axvline(x=best_epoch, color='green',
               linestyle='--', alpha=0.7,
               label=f'Best epoch ({best_epoch})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy Loss")
    ax.set_title(
        f"Cell 13: StoryReasoning Text Fine-tuning\n"
        f"{'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'} — {EXP_NAME}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "cell13_story_text_loss.png", experiment=EXP_FOLDER)
    plt.show()

    enc_tag = 'transformer' if USE_TRANSFORMER_ENCODER else 'lstm'
    save_checkpoint(
        text_ae, opt_story, N_STORY_EPOCHS, loss,
        filename = f"text_story_final_{enc_tag}.pth"
    )

    for p in text_ae.parameters():
        p.requires_grad = False

    frozen = sum(p.numel() for p in text_ae.parameters() if p.requires_grad)
    print(f"Text encoder frozen. Trainable params remaining: {frozen}")

    log_lines.append(f"Best epoch: {best_epoch}/{N_STORY_EPOCHS}")
    log_lines.append(f"Best val loss: {best_val_loss:.4f}")
    log_lines.append(f"Train initial: {story_losses[0]:.4f}")
    log_lines.append(f"Train final: {story_losses[-1]:.4f}")
    log_lines.append(f"Val initial: {story_val_losses[0]:.4f}")
    log_lines.append(f"Val final: {story_val_losses[-1]:.4f}")
    log_lines.append(
        f"Improvement: "
        f"{((story_val_losses[0]-story_val_losses[-1])/story_val_losses[0])*100:.1f}%"
    )
    log_lines.append("Text encoder frozen.")
    save_log(log_lines, experiment=EXP_FOLDER)

    print(f"Best epoch: {best_epoch}/{N_STORY_EPOCHS}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Train: {story_losses[0]:.4f} → {story_losses[-1]:.4f}")
    print(f"Val: {story_val_losses[0]:.4f} → {story_val_losses[-1]:.4f}")
def pretrain_visual():
    print("Visual Encoder Pretraining")
    print("=" * 55)
    print(f"Visual encoder: {'ResNet-18' if USE_RESNET_ENCODER else 'CNN baseline'}")
    print(f"Experiment: {EXP_NAME}\n")

    log_vis = [
        f"Experiment: {EXP_NAME}",
        f"Visual Encoder Pretraining",
        f"Visual encoder : {'ResNet-18' if USE_RESNET_ENCODER else 'CNN'}",
        "-" * 55,
    ]

    loss_fn_vis = nn.L1Loss()

    VIS_LR  = 0.0003 if USE_RESNET_ENCODER else 0.0005
    opt_vis = torch.optim.Adam(visual_ae.parameters(), lr=VIS_LR)

    print(f"Learning rate  : {VIS_LR}")
    print(f"(lower for ResNet to preserve ImageNet weights)\n"
          if USE_RESNET_ENCODER else
          f"(higher for CNN training from scratch)\n")
    print("Stage A: STL-10 External Pretraining")
    print("─" * 55)
    print("STL-10: 5000 images across 10 object categories.")

    stl_transform = transforms.Compose([
        transforms.Resize(IMAGE_HW),
        transforms.ToTensor(),
    ])

    stl_data = STL10(
        root= '/content/data',
        split= 'train',
        download = True,
        transform = stl_transform
    )

    stl_loader = DataLoader(
        stl_data,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers= 2,
        pin_memory = True
    )

    print(f"STL-10 loaded: {len(stl_data):,} images")
    print(f"Categories: {stl_data.classes}\n")

    N_STL_EPOCHS = 10
    MAX_BATCHES_STL = 200
    stl_losses = []

    sched_stl = torch.optim.lr_scheduler.StepLR(
        opt_vis, step_size=2, gamma=0.5
    )

    log_vis.append("Stage A — STL-10")
    log_vis.append(f"Epochs: {N_STL_EPOCHS}")
    log_vis.append(f"Max batches: {MAX_BATCHES_STL}")
    log_vis.append(f"LR: {VIS_LR}")
    log_vis.append("-" * 55)

    print(f"Training {N_STL_EPOCHS} epochs "
          f"(capped at {MAX_BATCHES_STL} batches/epoch)...\n")

    for epoch in range(N_STL_EPOCHS):
        visual_ae.train()
        epoch_loss= 0.0
        batch_count = 0
        t0 = time.time()

        for i, (imgs, _) in enumerate(stl_loader):
            if i >= MAX_BATCHES_STL:
                break
            imgs = imgs.to(device)

            opt_vis.zero_grad()
            x_content, x_context = visual_ae(imgs)

            loss = loss_fn_vis(x_content, imgs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(visual_ae.parameters(), 1.0)
            opt_vis.step()

            epoch_loss += loss.item()
            batch_count += 1

        sched_stl.step()
        avg = epoch_loss / batch_count
        elapsed = time.time() - t0
        stl_losses.append(avg)

        line = (f"Epoch {epoch+1}/{N_STL_EPOCHS} | "
                f"avg_loss={avg:.4f} | "
                f"last_loss={loss.item():.4f} | "
                f"time={elapsed:.1f}s")
        print(line)
        log_vis.append(line)

        save_checkpoint(
            visual_ae, opt_vis, epoch+1, loss,
            filename = f"visual_stl_epoch{epoch+1}_"
                       f"{'resnet' if USE_RESNET_ENCODER else 'cnn'}.pth"
        )

    log_vis.append(f"Stage A initial : {stl_losses[0]:.4f}")
    log_vis.append(f"Stage A final: {stl_losses[-1]:.4f}")
    log_vis.append("")

    print(f"\nStage A complete.")
    print(f"{stl_losses[0]:.4f} → {stl_losses[-1]:.4f}\n")

    print("Stage B: StoryReasoning Visual Fine-tuning")
    print("─" * 55)
    print("Fine-tunes on actual story frames.")
    print("Dual loss: content reconstruction + context consistency.\n")

    ae_ds = AutoEncoderTaskDataset(train_dataset)
    ae_loader = DataLoader(
        ae_ds,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers= 2,
        pin_memory = True
    )

    FINETUNE_VIS_LR = VIS_LR * 0.5
    for pg in opt_vis.param_groups:
        pg['lr'] = FINETUNE_VIS_LR

    sched_story_vis = torch.optim.lr_scheduler.StepLR(
        opt_vis, step_size=2, gamma=0.5
    )

    N_VIS_FINETUNE = 5
    story_vis_losses = []
    best_vis_loss = float('inf')
    best_vis_epoch = 0

    log_vis.append("Stage B - StoryReasoning Fine-tuning")
    log_vis.append(f"Epochs: {N_VIS_FINETUNE}")
    log_vis.append(f"LR: {FINETUNE_VIS_LR}")
    log_vis.append("-" * 55)

    print(f"Fine-tuning {N_VIS_FINETUNE} epochs on StoryReasoning \n")

    for epoch in range(N_VIS_FINETUNE):
        visual_ae.train()
        epoch_loss = 0.0
        batch_count = 0
        t0 = time.time()

        for (imgs,) in ae_loader:
            imgs = imgs.to(device)

            opt_vis.zero_grad()
            x_content, x_context = visual_ae(imgs)

            loss_content = loss_fn_vis(x_content, imgs)

            mean_img = imgs.mean(dim=0, keepdim=True).expand_as(x_context)
            loss_context = loss_fn_vis(x_context, mean_img)

            loss = loss_content + 0.5 * loss_context

            loss.backward()
            torch.nn.utils.clip_grad_norm_(visual_ae.parameters(), 1.0)
            opt_vis.step()

            epoch_loss += loss.item()
            batch_count += 1

        sched_story_vis.step()
        avg = epoch_loss / batch_count
        elapsed = time.time() - t0
        story_vis_losses.append(avg)

        if avg < best_vis_loss:
            best_vis_loss = avg
            best_vis_epoch = epoch + 1
            save_checkpoint(
                visual_ae, opt_vis, epoch+1, loss,
                filename = f"visual_story_best_"
                           f"{'resnet' if USE_RESNET_ENCODER else 'cnn'}.pth"
            )

        line = (f"Epoch {epoch+1}/{N_VIS_FINETUNE} | "
                f"avg_loss={avg:.4f} | "
                f"last_loss={loss.item():.4f} | "
                f"time={elapsed:.1f}s"
                f"{'<- best' if epoch+1 == best_vis_epoch else ''}")
        print(line)
        log_vis.append(line)

        save_checkpoint(
            visual_ae, opt_vis, epoch+1, loss,
            filename = f"visual_story_epoch{epoch+1}_"
                       f"{'resnet' if USE_RESNET_ENCODER else 'cnn'}.pth"
        )

    print(f"\nLoading best checkpoint "
          f"(epoch {best_vis_epoch}, loss={best_vis_loss:.4f})")
    load_checkpoint(
        visual_ae,
        filename = f"visual_story_best_"
                   f"{'resnet' if USE_RESNET_ENCODER else 'cnn'}.pth"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        range(1, N_STL_EPOCHS+1), stl_losses,
        marker='o', linewidth=2,
        color='steelblue', label='STL-10'
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 Loss")
    axes[0].set_title(
        f"Stage A: STL-10 Pretraining\n"
        f"({'ResNet-18' if USE_RESNET_ENCODER else 'CNN'})"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        range(1, N_VIS_FINETUNE+1), story_vis_losses,
        marker='s', linewidth=2,
        color='darkorange', label='StoryReasoning'
    )
    axes[1].axvline(
        x=best_vis_epoch, color='green',
        linestyle='--', alpha=0.7,
        label=f'Best epoch ({best_vis_epoch})'
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("L1 Loss")
    axes[1].set_title(
        f"Stage B: StoryReasoning Fine-tuning\n"
        f"({'ResNet-18' if USE_RESNET_ENCODER else 'CNN'})"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Visual Encoder Pretraining — {EXP_NAME}",
        fontsize=13
    )
    plt.tight_layout()
    save_figure(fig, "cell14_visual_pretraining_loss.png", experiment=EXP_FOLDER)
    plt.show()

    vis_tag = 'resnet' if USE_RESNET_ENCODER else 'cnn'
    save_checkpoint(
        visual_ae, opt_vis, N_VIS_FINETUNE, loss,
        filename = f"visual_pretrained_final_{vis_tag}.pth"
    )

    log_vis.append("")
    log_vis.append("-" * 55)
    log_vis.append(f"STL-10 initial: {stl_losses[0]:.4f}")
    log_vis.append(f"STL-10 final: {stl_losses[-1]:.4f}")
    log_vis.append(f"StoryReasoning init : {story_vis_losses[0]:.4f}")
    log_vis.append(f"StoryReasoning final : {story_vis_losses[-1]:.4f}")
    log_vis.append(f"Best epoch : {best_vis_epoch}/{N_VIS_FINETUNE}")
    log_vis.append(f"Best loss : {best_vis_loss:.4f}")
    log_vis.append(
        f"Improvement          : "
        f"{((stl_losses[0]-story_vis_losses[-1])/stl_losses[0])*100:.1f}% "
        f"(STL-10 start -> StoryReasoning best)"
    )
    log_vis.append("Visual encoder ready.")
    save_log(log_vis, experiment=EXP_FOLDER)

    print(f"STL-10: {stl_losses[0]:.4f} → {stl_losses[-1]:.4f}")
    print(f"StoryReasoning: {story_vis_losses[0]:.4f} → "
          f"{story_vis_losses[-1]:.4f}")
    print(f"Best epoch: {best_vis_epoch}/{N_VIS_FINETUNE}")

def train_main():
    print("Main Training - SequencePredictor")
    print("=" * 55)
    print(f"Experiment: {EXP_NAME}")
    print(f"Text encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}")
    print(f"Visual encoder: {'ResNet-18' if USE_RESNET_ENCODER else 'CNN'}")
    print(f"Contrastive loss : {'ON' if USE_CONTRASTIVE_ROI else 'OFF'}")
    print(f"Device : {device}\n")

    for p in seq_pred.parameters():
        p.requires_grad = True

    total_train = sum(p.numel() for p in seq_pred.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_train:,}\n")

    crit_img = nn.L1Loss()
    crit_ctx   = nn.MSELoss()
    crit_text = nn.CrossEntropyLoss(
        ignore_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )
    opt_main = torch.optim.Adam(seq_pred.parameters(), lr=LR)
    sched_main = torch.optim.lr_scheduler.StepLR(
        opt_main, step_size=3, gamma=0.5
    )

    main_ds = SequencePredictionDataset(train_dataset, tokenizer)

    val_size = int(0.1 * len(main_ds))
    train_size = len(main_ds) - val_size

    train_data, val_data = random_split(
        main_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Main training samples: {len(train_data):,}")
    print(f"Main validation samples: {len(val_data):,}\n")

    log_main = [
        f"Experiment: {EXP_NAME}",
        f" Main Training",
        f"Text encoder: {'Transformer' if USE_TRANSFORMER_ENCODER else 'LSTM'}",
        f"Visual encoder: {'ResNet-18' if USE_RESNET_ENCODER else 'CNN'}",
        f"Contrastive loss : {'ON' if USE_CONTRASTIVE_ROI else 'OFF'}",
        f"Epochs: {N_EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"LR: {LR}",
        f"Trainable params : {total_train:,}",
        "-" * 55,
    ]

    train_losses = []
    val_losses= []
    best_val= float('inf')
    best_ep = 0

    print(f"Training for {N_EPOCHS} epochs\n")

    for epoch in range(N_EPOCHS):
        seq_pred.train()
        epoch_loss= 0.0
        batch_count = 0
        t0= time.time()

        for (frames, descs, img_tgt, txt_tgt,
             roi1, roi2, roi_valid, roi_frame, ent_id) in train_loader:

            frames = frames.to(device)
            descs = descs.to(device)
            img_tgt = img_tgt.to(device)
            txt_tgt = txt_tgt.to(device)
            roi1 = roi1.to(device)
            roi2 = roi2.to(device)
            roi_valid = roi_valid.to(device)
            roi_frame = roi_frame.to(device)
            opt_main.zero_grad()

            pred_img, pred_ctx, pred_text, h0, c0, z_v_seq, z_t_seq = seq_pred(
                frames, descs, txt_tgt
            )
            loss_img = crit_img(pred_img, img_tgt)

            mu_global= frames.mean(dim=[0,1]).unsqueeze(0).expand_as(pred_ctx)
            loss_ctx = crit_ctx(pred_ctx, mu_global)

            pred_flat = pred_text.reshape(-1, tokenizer.vocab_size)
            tgt_flat= txt_tgt.squeeze(1)[:, 1:].reshape(-1)
            loss_text = crit_text(pred_flat, tgt_flat)

            loss_contrast = torch.tensor(0.0, device=device)
            loss_reid= torch.tensor(0.0, device=device)

            if USE_CONTRASTIVE_ROI and roi_valid.any():
                mask = roi_valid.bool()
                if mask.sum() > 1:
                    z_roi1 = seq_pred.image_encoder(roi1[mask])
                    z_roi2 = seq_pred.image_encoder(roi2[mask])

                    loss_reid = F.mse_loss(z_roi1, z_roi2)

                    if USE_FRAME_AWARE_GROUNDING:
                        rf = roi_frame[mask].clamp(0, z_t_seq.size(1)-1)
                        z_txt= z_t_seq[mask][
                            torch.arange(mask.sum()), rf
                        ]
                    else:
                        z_txt = z_t_seq[mask, 0, :]

                    z_roi_n = F.normalize(z_roi1, dim=-1)
                    z_txt_n = F.normalize(z_txt,  dim=-1)
                    sim = torch.mm(z_roi_n, z_txt_n.T) / CONTRASTIVE_TAU
                    labels = torch.arange(sim.size(0), device=device)
                    loss_contrast = F.cross_entropy(sim, labels)

            total_loss = (
                loss_img
                + loss_ctx
                + loss_text
                + LAMBDA_REID     * loss_reid
                + LAMBDA_CONTRAST * loss_contrast
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(seq_pred.parameters(), 1.0)
            opt_main.step()

            epoch_loss  += total_loss.item()
            batch_count += 1

        avg_train = epoch_loss / batch_count
        seq_pred.eval()
        val_loss    = 0.0
        val_batches = 0

        with torch.no_grad():
            for (frames, descs, img_tgt, txt_tgt, *rest) in val_loader:
                frames  = frames.to(device)
                descs= descs.to(device)
                img_tgt = img_tgt.to(device)
                txt_tgt = txt_tgt.to(device)

                pred_img, pred_ctx, pred_text, _, _, _, _ = seq_pred(
                    frames, descs, txt_tgt
                )
                v_img  = crit_img(pred_img, img_tgt)
                v_text = crit_text(
                    pred_text.reshape(-1, tokenizer.vocab_size),
                    txt_tgt.squeeze(1)[:, 1:].reshape(-1)
                )
                val_loss   += (v_img + v_text).item()
                val_batches += 1

        avg_val = val_loss / val_batches
        elapsed = time.time() - t0

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        sched_main.step()

        if avg_val < best_val:
            best_val = avg_val
            best_ep = epoch + 1
            save_checkpoint(
                seq_pred, opt_main, epoch+1, total_loss,
                filename=f"seq_pred_best_{EXP_NAME}.pth"
            )

        line = (f"Epoch {epoch+1}/{N_EPOCHS} | "
                f"train={avg_train:.4f} | "
                f"val={avg_val:.4f} | "
                f"time={elapsed:.1f}s"
                f"{'  <- best' if epoch+1 == best_ep else ''}")
        print(line)
        log_main.append(line)

    fig, ax = plt.subplots(figsize=(10, 5))
    epochs= range(1, N_EPOCHS+1)
    ax.plot(epochs, train_losses,
            marker='o', linewidth=2,
            color='steelblue', label='Train Loss')
    ax.plot(epochs, val_losses,
            marker='s', linewidth=2,
            color='darkorange', label='Val Loss')
    ax.axvline(x=best_ep, color='green',
               linestyle='--', alpha=0.7,
               label=f'Best epoch ({best_ep})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title(f"Main Training — {EXP_NAME}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, "cell15_main_training_loss.png", experiment=EXP_FOLDER)
    plt.show()

    save_checkpoint(
        seq_pred, opt_main, N_EPOCHS, total_loss,
        filename=f"seq_pred_final_{EXP_NAME}.pth"
    )

    log_main.append("-" * 55)
    log_main.append(f"Best epoch: {best_ep}/{N_EPOCHS}")
    log_main.append(f"Best val loss: {best_val:.4f}")
    log_main.append(f"Train initial: {train_losses[0]:.4f}")
    log_main.append(f"Train final: {train_losses[-1]:.4f}")
    log_main.append(f"Val initial: {val_losses[0]:.4f}")
    log_main.append(f"Val final: {val_losses[-1]:.4f}")
    log_main.append(
        f"Improvement: "
        f"{( (val_losses[0]-val_losses[-1])/val_losses[0] )*100:.1f}%"
    )
    save_log(log_main, experiment=EXP_FOLDER)

    print(f"Best epoch: {best_ep}/{N_EPOCHS}")
    print(f"Best val: {best_val:.4f}")
    print(f"Train: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    print(f"Val: {val_losses[0]:.4f} -> {val_losses[-1]:.4f}")
    print(f"\n Explainability.")
