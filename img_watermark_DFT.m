function img_watermark_DFT(img1_path,img2_path)
    
%% Eliminate all warnings
    warning('off');
    
%% Set the value when embedding the watermark
    alpha=1;

%% Read the image and show them
    im=im2double(imread(img1_path));
    mark=im2double(imread(img2_path));
    figure, imshow(im),title('The original image');
    figure, imshow(mark),title('The watermark');

%% Encode the watermark
    %Get the original image size
    imsize = size(im);

    % random:The coding method adopts random sequence coding, 
    %        through coding, the watermark is distributed to random 
    %        distribution to each frequency
    TH=zeros(imsize(1)*0.5,imsize(2),imsize(3));
    TH1 = TH;
    TH1(1:size(mark,1),1:size(mark,2),:) = mark;
    M=randperm(0.5*imsize(1));
    N=randperm(imsize(2));
    save('encode.mat','M','N');
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            TH(i,j,:)=TH1(M(i),N(j),:);
        end
    end

    % The watermark is encrypted, and the watermark is symmetrical.
    watmark_encoded = zeros(imsize(1),imsize(2),imsize(3));
    watmark_encoded(1:imsize(1)*0.5,1:imsize(2),:)=TH;
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watmark_encoded(imsize(1)+1-i,imsize(2)+1-j,:)=TH(i,j,:);
        end
    end

    figure,imshow(watmark_encoded),title('encoded watermark');
    
%% Add watermark
    % Because the image is a discrete signal, the discrete Fourier transform is used.
    % The two-dimensional fast Fourier transform used in this code, whcih is equivalent to discrete Fourier transform
    
    ori_img_fft=fft2(im); % Get the spectrum of the original picture
    watermarked_img_fft=ori_img_fft+alpha*double(watmark_encoded); % Overlay the spectrum of original image and the watermark
    watermarked_img=ifft2(watermarked_img_fft); % Get the watermarked image through the inverse Fourier transform
    residual_ori_embedded = watermarked_img-double(im); % Get the spectrum difference before and after adding the watermark

    % Draw the key image
    figure,imshow(ori_img_fft);title('spectrum of original image');
    figure,imshow(watermarked_img_fft); title('spectrum of watermarked image');
    figure,imshow(watermarked_img);title('watermarked image');
    figure,imshow(uint8(residual_ori_embedded)); title('residual');

%% Extract watermark
    % Perform Fourier transform on the watermarked image to get the spectrum
    % Obtain the spectrum of the watermark by the difference of the spectrum of image before and after watermarked
    watermarked_img_fft2=fft2(watermarked_img);
    G=(watermarked_img_fft2-ori_img_fft)/alpha;
    watermark_extracted=G;
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted(M(i),N(j),:)=G(i,j,:);
        end
    end
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted(imsize(1)+1-i,imsize(2)+1-j,:)=watermark_extracted(i,j,:);
        end
    end
    figure,imshow(watermark_extracted);title('extracted watermark');

%% Robustness test: adding noise
    watermarked_img_noise=imnoise(watermarked_img,'salt & pepper');
    figure,imshow(watermarked_img_noise);title('The watermarked img adding noise');
    watermarked_img_cropping_fft2=fft2(watermarked_img_noise);
    G=(watermarked_img_cropping_fft2-ori_img_fft)/alpha;
    watermark_extracted_noise=G;
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_noise(M(i),N(j),:)=G(i,j,:);
        end
    end
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_noise(imsize(1)+1-i,imsize(2)+1-j,:)=watermark_extracted_noise(i,j,:);
        end
    end

    figure,imshow(watermark_extracted_noise);title('extracted watermark noise');

%% Robustness test: cropping
    watermarked_img_cropping=im2uint8(watermarked_img);
    watermarked_img_cropping_r= watermarked_img_cropping(:,:,1);
    watermarked_img_cropping_r(1:320,1:796.5)=255;
    watermarked_img_cropping_g =watermarked_img_cropping(:,:,2);
    watermarked_img_cropping_g(1:320,1:796.5)=255;
    watermarked_img_cropping_b =watermarked_img_cropping(:,:,3);
    watermarked_img_cropping_b(1:320,1:796.5)=255;
    watermarked_img_cropping(:,:,1) = watermarked_img_cropping_r;
    watermarked_img_cropping(:,:,2) = watermarked_img_cropping_g;
    watermarked_img_cropping(:,:,3) = watermarked_img_cropping_b;
    figure,imshow(watermarked_img_cropping);title('watermarked_img_cropping');
    watermarked_img_cropping_fft2=fft2(im2double(watermarked_img_cropping));
    G=(watermarked_img_cropping_fft2-ori_img_fft)/alpha;
    watermark_extracted_cropping=G;
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_cropping(M(i),N(j),:)=G(i,j,:);
        end
    end
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_cropping(imsize(1)+1-i,imsize(2)+1-j,:)=watermark_extracted_cropping(i,j,:);
        end
    end

    figure,imshow(watermark_extracted_cropping);title('extracted watermark cropping');

%% Robustness test: rotateing
    watermarked_img_rotateing=imrotate(im2double(watermarked_img),15,'nearest','crop');
    figure,imshow(watermarked_img_rotateing);title('watermarked_img_rotateing');
    watermarked_img_rotateing_fft2=fft2(watermarked_img_cropping);
    G=(watermarked_img_rotateing_fft2-ori_img_fft)/alpha;
    watermark_extracted_rotateing=G;
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_rotateing(M(i),N(j),:)=G(i,j,:);
        end
    end
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_rotateing(imsize(1)+1-i,imsize(2)+1-j,:)=watermark_extracted_rotateing(i,j,:);
        end
    end

    figure,imshow(watermark_extracted_rotateing);title('extracted watermark rotateing');
    
%% Robustness test: brightness increased by 10%
    watermarked_img_brightness10=1.1*im2double(watermarked_img);
    figure,imshow(watermarked_img_brightness10);title('watermarked_img_brightness10');
    watermarked_img_brightness_fft2=fft2(watermarked_img_brightness10);
    G=(watermarked_img_brightness_fft2-ori_img_fft)/alpha;
    watermark_extracted_brightness=G;
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_brightness(M(i),N(j),:)=G(i,j,:);
        end
    end
    for i=1:imsize(1)*0.5
        for j=1:imsize(2)
            watermark_extracted_brightness(imsize(1)+1-i,imsize(2)+1-j,:)=watermark_extracted_brightness(i,j,:);
        end
    end

    figure,imshow(watermark_extracted_brightness);title('extracted watermark brightness10');