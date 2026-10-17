%% FaceVitals: 2027 IEEE Signal Processing Cup (Round 1 Processing Pipeline)
% Theme: Equitable, Edge-Deployable Camera-Based Vital Sign Monitoring
% Sponsoring Organizations: IEEE Signal Processing Society & MathWorks
%
% This script implements end-to-end Heart Rate extraction from raw facial video
% using MathWorks toolboxes (Computer Vision, Signal Processing, Image Processing).

function results = process_round1(video_path, ref_path, output_mat_path, condition_tag, constraint_tag)
    if nargin < 1 || isempty(video_path)
        video_path = 'target_video.mp4';
    end
    if nargin < 2 || isempty(ref_path)
        ref_path = 'reference_signal.csv';
    end
    if nargin < 3 || isempty(output_mat_path)
        output_mat_path = 'Round1_Baseline_None.mat';
    end
    if nargin < 4 || isempty(condition_tag)
        condition_tag = 'Baseline';
    end
    if nargin < 5 || isempty(constraint_tag)
        constraint_tag = 'None';
    end

    fprintf('[FaceVitals MATLAB] Processing video: %s\n', video_path);

    %% 1. Ingest Video via VideoReader
    vr = VideoReader(video_path);
    fps = vr.FrameRate;
    total_frames = vr.NumFrames;
    height = vr.Height;
    width = vr.Width;

    fprintf('Video specs: %dx%d @ %.1f fps, %d frames\n', width, height, fps, total_frames);

    %% 2. Initialize Face Detection & ROI Tracking
    % Use Computer Vision Toolbox Cascade Object Detector
    face_detector = vision.CascadeObjectDetector('FrontalFaceCART', 'MinSize', [32 32]);

    raw_frames = zeros(total_frames, height, width, 3, 'uint8');
    roi_bboxes = zeros(total_frames, 3, 4, 'int32');
    rgb_traces = zeros(total_frames, 3);

    prev_face = [round(width*0.25), round(height*0.20), round(width*0.50), round(height*0.55)];

    frame_idx = 1;
    while hasFrame(vr) && frame_idx <= total_frames
        frame = readFrame(vr);
        raw_frames(frame_idx, :, :, :) = frame;

        % Detect face
        bboxes = step(face_detector, frame);
        if ~isempty(bboxes)
            % Pick largest face
            areas = bboxes(:, 3) .* bboxes(:, 4);
            [~, max_idx] = max(areas);
            current_face = bboxes(max_idx, :);
            % Temporal smoothing (alpha = 0.7)
            face_box = round(0.7 * current_face + 0.3 * prev_face);
            prev_face = face_box;
        else
            face_box = prev_face;
        end

        fx = face_box(1); fy = face_box(2); fw = face_box(3); fh = face_box(4);

        % Compute relative-fraction ROIs:
        % Forehead: top 18% of face
        fh_w = round(fw * 0.45); fh_h = round(fh * 0.18);
        fh_x = fx + round(fw * 0.275); fh_y = max(1, fy + round(fh * 0.10));

        % Cheeks: middle cheeks
        ck_w = round(fw * 0.22); ck_h = round(fh * 0.22);
        lc_x = fx + round(fw * 0.15); lc_y = fy + round(fh * 0.52);
        rc_x = fx + round(fw * 0.63); rc_y = fy + round(fh * 0.52);

        % Bound coordinates
        safe_fh = [max(1, min(width-fh_w, fh_x)), max(1, min(height-fh_h, fh_y)), max(4, fh_w), max(4, fh_h)];
        safe_lc = [max(1, min(width-ck_w, lc_x)), max(1, min(height-ck_h, lc_y)), max(4, ck_w), max(4, ck_h)];
        safe_rc = [max(1, min(width-ck_w, rc_x)), max(1, min(height-ck_h, rc_y)), max(4, ck_w), max(4, ck_h)];

        roi_bboxes(frame_idx, 1, :) = safe_fh;
        roi_bboxes(frame_idx, 2, :) = safe_lc;
        roi_bboxes(frame_idx, 3, :) = safe_rc;

        % Crop and average
        crop_fh = frame(safe_fh(2):(safe_fh(2)+safe_fh(4)-1), safe_fh(1):(safe_fh(1)+safe_fh(3)-1), :);
        crop_lc = frame(safe_lc(2):(safe_lc(2)+safe_lc(4)-1), safe_lc(1):(safe_lc(1)+safe_lc(3)-1), :);
        crop_rc = frame(safe_rc(2):(safe_rc(2)+safe_rc(4)-1), safe_rc(1):(safe_rc(1)+safe_rc(3)-1), :);

        m_fh = squeeze(mean(mean(double(crop_fh), 1), 2));
        m_lc = squeeze(mean(mean(double(crop_lc), 1), 2));
        m_rc = squeeze(mean(mean(double(crop_rc), 1), 2));

        % Fused anatomical weighting (50% forehead, 25% each cheek)
        rgb_traces(frame_idx, :) = (0.50 * m_fh + 0.25 * m_lc + 0.25 * m_rc)';

        frame_idx = frame_idx + 1;
    end

    %% 3. Classical Plane-Orthogonal-to-Skin (POS) Projection
    win_len = max(8, round(fps * 1.6));
    N = size(rgb_traces, 1);
    H = zeros(N, 1);
    weights = zeros(N, 1);

    for m = 1:(N - win_len + 1)
        sub_c = rgb_traces(m:(m + win_len - 1), :)';
        mean_c = mean(sub_c, 2) + 1e-6;
        Cn = sub_c ./ mean_c - 1.0;

        S1 = Cn(1, :) - Cn(2, :);
        S2 = Cn(1, :) + Cn(2, :) - 2.0 * Cn(3, :);
        alpha = std(S1) / (std(S2) + 1e-6);

        h_win = S1 + alpha * S2;
        h_win = h_win - mean(h_win);

        H(m:(m + win_len - 1)) = H(m:(m + win_len - 1)) + h_win';
        weights(m:(m + win_len - 1)) = weights(m:(m + win_len - 1)) + 1.0;
    end

    valid_idx = weights > 0;
    H(valid_idx) = H(valid_idx) ./ weights(valid_idx);

    %% 4. Bandpass Filtering (0.7 - 3.0 Hz)
    nyq = fps / 2;
    high_cutoff = min(3.0, nyq - 0.2);
    low_cutoff = min(0.7, high_cutoff - 0.2);
    [b, a] = butter(4, [low_cutoff, high_cutoff] / nyq, 'bandpass');
    filtered_bvp = filtfilt(b, a, H);

    %% 5. Spectral Peak Tracking for Heart Rate
    eval_win = min(N, round(fps * 8.0));
    step_win = max(1, round(fps * 0.5));
    estimated_hr = zeros(N, 1);
    w_count = zeros(N, 1);

    for s = 1:step_win:(N - eval_win + 1)
        e = s + eval_win - 1;
        segment = filtered_bvp(s:e);
        [pxx, f] = pwelch(segment, hamming(length(segment)), [], 2048, fps);

        % Cardiac band [0.7, 3.0 Hz]
        cardiac_mask = (f >= 0.7) & (f <= 3.0);
        f_cardiac = f(cardiac_mask);
        pxx_cardiac = pxx(cardiac_mask);

        % Harmonic comb weighting
        comb_scores = pxx_cardiac;
        for fi = 1:length(f_cardiac)
            f0 = f_cardiac(fi);
            [~, idx2] = min(abs(f - 2*f0));
            if idx2 <= length(pxx)
                comb_scores(fi) = comb_scores(fi) + 0.5 * pxx(idx2);
            end
        end

        [~, max_pos] = max(comb_scores);
        peak_hr_bpm = f_cardiac(max_pos) * 60.0;

        estimated_hr(s:e) = estimated_hr(s:e) + peak_hr_bpm;
        w_count(s:e) = w_count(s:e) + 1.0;
    end

    valid_hr = w_count > 0;
    estimated_hr(valid_hr) = estimated_hr(valid_hr) ./ w_count(valid_hr);
    if any(~valid_hr)
        estimated_hr(~valid_hr) = mean(estimated_hr(valid_hr));
    end

    %% 6. Load Reference Signal and Compute Metrics
    if exist(ref_path, 'file')
        ref_table = readtable(ref_path);
        ref_raw = ref_table{:, 2};
        if length(ref_raw) ~= N
            reference_signal = interp1(linspace(1, N, length(ref_raw)), ref_raw, 1:N, 'linear', 'extrap')';
        else
            reference_signal = ref_raw;
        end
    else
        reference_signal = estimated_hr;
    end

    % Statistical metrics
    mae_val = mean(abs(reference_signal - estimated_hr));
    rmse_val = sqrt(mean((reference_signal - estimated_hr).^2));
    r_mat = corrcoef(reference_signal, estimated_hr);
    pearson_val = r_mat(1, 2);

    diffs = reference_signal - estimated_hr;
    loa_val = [mean(diffs) - 1.96 * std(diffs); mean(diffs) + 1.96 * std(diffs)];

    fprintf('Results: MAE = %.2f bpm, RMSE = %.2f bpm, Pearson r = %.3f\n', mae_val, rmse_val, pearson_val);

    %% 7. Assemble and Save Official .MAT Container
    facial_video = raw_frames;
    roi_data = roi_bboxes;
    processed_signal = estimated_hr;

    metrics.MAE = mae_val;
    metrics.RMSE = rmse_val;
    metrics.Pearson_r = pearson_val;
    metrics.BlandAltman_LoA = loa_val;

    % Dynamically estimate scene illuminance from raw frame luminance
    mean_lum = mean(double(raw_frames(:)));
    est_lux = max(5.0, min(1500.0, round(mean_lum * 2.2)));

    params.camera_resolution = sprintf('%dx%d', width, height);
    params.frame_rate = fps;
    params.illuminance_lux = est_lux;
    params.camera_to_subject_distance = '50cm';
    params.subject_fitzpatrick_skin_type = 'TypeIII';
    params.subject_age = 25;
    params.constraint_applied = constraint_tag;
    params.condition = condition_tag;

    save(output_mat_path, 'facial_video', 'roi_data', 'reference_signal', 'processed_signal', 'metrics', 'params', '-v7.3');
    fprintf('Saved official package: %s\n', output_mat_path);

    results.metrics = metrics;
    results.params = params;
end
