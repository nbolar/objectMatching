%% CS413 Coursework Assignment 2017 - Object Matching in Videos
%
% Most of the code used in this project is a combination of examples provided in the
% laboratory as well as my own understanding of the material taught. Some
% bits of the code is attributed to the multiple online MATLAB(R) tutorials
% on object detection available on their website. [https://uk.mathworks.com/help/vision/examples/object-detection-in-a-cluttered-scene-using-point-feature-matching.html]


clear all
clc
warning off;

im1 = (imread('objects-1.jpeg'));
im2 = (imread('objects-2.jpeg'));
im3 = (imread('objects-3.jpeg'));

v1 = VideoReader('objects-test-1.mov');
v2 = VideoReader('objects-test-2.mov');
v3 = VideoReader('objects-test-3.mov');
v4 = VideoReader('objects-test-4.mov');


%% Video Properties

vid1width = v1.Width;
vid1height = v1.Height;
vid1_frames = v1.NumberOfFrames;

vid2width = v2.Width;
vid2height = v2.Height;
vid2_frames = v2.NumberOfFrames;

vid3width = v3.Width;
vid3height = v3.Height;
vid3_frames = v3.NumberOfFrames;

vid4width = v4.Width;
vid4height = v4.Height;
vid4_frames = v4.NumberOfFrames;


%% SURF Feature Extraction, Matching and Object Detection for Video 1 -> objects-test-1.mov


points1 = detectSURFFeatures(rgb2gray(im1),'MetricThreshold',800);
points2 = detectSURFFeatures(rgb2gray(im2),'MetricThreshold',800);



[fp1, vp1] = extractFeatures(rgb2gray(im1), points1, 'SURFSize', 64);           %SURF Feature Descriptors
[fp2, vp2] = extractFeatures(rgb2gray(im2), points2, 'SURFSize', 64);           %SURF Feature Descriptors

figure('Position',[720, 720, 720, 1080]);
status1 = 0;
flag1 = 0;
check1 = 0;
for n = 1: ((vid1_frames / 4) -4)
    thisFrame = imresize(read(v1,4*n),1.5);
    points_v1 = detectSURFFeatures(rgb2gray(thisFrame),'MetricThreshold',1000);
    [fpv1, vpv1] = extractFeatures(rgb2gray(thisFrame), points_v1, 'SURFSize', 64);
    pointspairs1 = matchFeatures(fp1,fpv1,'Unique',true);                       %Matching Features between the first training image and the video frame
    matchedPointsPairs1 = vp1(pointspairs1(:,1),:);
    matchedScenePoints1 = vpv1(pointspairs1(:,2),:);
    
    pointspairs2 = matchFeatures(fp2,fpv1,'Unique',true);                      %Matching Features between the second training image and the video frame
    matchedPointsPairs2 = vp2(pointspairs2(:,1),:);
    matchedScenePoints2 = vpv1(pointspairs2(:,2),:);
    
   
    if length(pointspairs2) < length(pointspairs1)                              %Metric used to verify the similarity of a video frame to its corresponding training image
        if (size(pointspairs1,1) >=10)                                          %Used to remove outliers apart from the methods implemented below
            [tform1, inlierPointsPairs1, inlierScenePoints1,status1] = estimateGeometricTransform(matchedPointsPairs1, matchedScenePoints1, 'projective'); 
            if (inlierPointsPairs1.Count <=5)
                flag1 = -1;
                subplot(2,1,2)
                imshow(thisFrame);
            else
                flag1 = 1;
                check1 = 1;
                subplot(2,1,2)
                percentage = (length(inlierPointsPairs1) / length(pointspairs1)) * 100;
                text_input = ['Confidence: ' num2str(percentage,'%0.2f') '%'', Thresh +/- = 1000, Octaves = 3'];
                text = insertText(thisFrame,[230 680],text_input,'FontSize',22,'BoxColor','blue','BoxOpacity',0.4,'TextColor','white');         %Labeling the recognised Object
                corners = [0,0;960,0;960,720;0,720];
                imshow(text);                                                 %Locating the recognised object
                new_corners = transformPointsForward(tform1, corners);
                hold on;
                patch(new_corners(:,1),new_corners(:,2),[0 1 0],'FaceAlpha',0.5);       %Drawing a box around the recognised object
                title('Object Recognised','interpreter','latex','fontsize',16);
                hold off;
            end
        else
            flag1 = -1;
            subplot(2,1,2)
            imshow(thisFrame);
        end
    else
        if (size(pointspairs2,1) >=10)                                       %Used to remove outliers apart from the methods implemented below
        [tform2, inlierPointsPairs1, inlierScenePoints1,status1] = estimateGeometricTransform(matchedPointsPairs2, matchedScenePoints2, 'projective'); 
        
            if (inlierPointsPairs1.Count <=5)
                flag1 = -2;
                subplot(2,1,2)
                imshow(thisFrame);
            else
                flag1 = 2;
                check1 = 2;
                subplot(2,1,2)
                percentage = (length(inlierPointsPairs1) / length(pointspairs2)) * 100;
                text_input = ['Confidence: ' num2str(percentage,'%0.2f') '%'', Thresh +/- = 1000, Octaves = 3'];
                text = insertText(thisFrame,[230 680],text_input,'FontSize',22,'BoxColor','blue','BoxOpacity',0.4,'TextColor','white');     %Labeling the recognised Object
                corners = [0,0;960,0;960,720;0,720];
                new_corners = transformPointsForward(tform2, corners);
                imshow(text);                                               %Locating the recognised object
                hold on;
                patch(new_corners(:,1),new_corners(:,2),[0 1 0],'FaceAlpha',0.5);       %Drawing a box around the recognised object
                title('Object Recognised','interpreter','latex','fontsize',16);
                hold off;
            end
        else
            flag1 = -2;
            subplot(2,1,2)
            imshow(thisFrame);
 
        end
    end
          
    if flag1 == 1
        subplot(2,1,1)
        showMatchedFeatures(im1, thisFrame, inlierPointsPairs1, inlierScenePoints1, 'montage');     %Shows the matching features montage on the same figure as the detection method
% 
    elseif flag1 == 2
        subplot(2,1,1)
        showMatchedFeatures(im2, thisFrame, inlierPointsPairs1, inlierScenePoints1, 'montage');
% 
    elseif (flag1 == -1 && check1 ~=2 )
        subplot(2,1,1)
        showMatchedFeatures(im1, thisFrame, [nan nan], [nan nan], 'montage');                   %Method used to neglect false positive inlier points
% 
    elseif (flag1 == -2 && check1 ~=1)
        subplot(2,1,1)
        showMatchedFeatures(im2, thisFrame, [nan nan], [nan nan], 'montage');
% 
    elseif (flag1 == -2 && check1 ==1)
        subplot(2,1,1)
        showMatchedFeatures(im1, thisFrame, [nan nan], [nan nan], 'montage');
% 
    elseif (flag1 == -1 && check1 ==2)
        subplot(2,1,1)
        showMatchedFeatures(im2, thisFrame, [nan nan], [nan nan], 'montage');
% 
    end      
    
end


%% SURF Feature Extraction, Matching and Object Detection for Video 2 (Cluttered) -> objects-test-2.mov


clear pointspairs1 matchedPointsPairs1 matchedScenePoints1 pointspairs2 matchedPointsPairs2 matchedScenePoints2 percentage
clear tform1 inlierPointsPairs1 inlierScenePoints1 tform2 inlierPointsPairs1 inlierScenePoints1
clear points1 points2 fp1 vp1 fp2 vp2

points1 = detectSURFFeatures(rgb2gray(im1),'MetricThreshold',1000);
points2 = detectSURFFeatures(rgb2gray(im2),'MetricThreshold',1000);



[fp1, vp1] = extractFeatures(rgb2gray(im1), points1, 'SURFSize', 64);
[fp2, vp2] = extractFeatures(rgb2gray(im2), points2, 'Method', 'SURF');

figure('Position',[720, 720, 720, 1080]);
status2 = 0;
flag2 = 0;
check2 = 0;
for n = 1: ((vid2_frames / 4) -4)
    thisFrame = imresize(read(v2,4*n),1.5);
    points_v2 = detectSURFFeatures(rgb2gray(thisFrame),'MetricThreshold',800);
    [fpv2, vpv2] = extractFeatures(rgb2gray(thisFrame), points_v2, 'Method', 'SURF');
    pointspairs1 = matchFeatures(fp1,fpv2,'Unique',true);                       %Matching Features between the first training image and the video frame
    matchedPointsPairs1 = vp1(pointspairs1(:,1),:);
    matchedScenePoints1 = vpv2(pointspairs1(:,2),:);
    
    pointspairs2 = matchFeatures(fp2,fpv2,'Unique',true);                       %Matching Features between the second training image and the video frame
    matchedPointsPairs2 = vp2(pointspairs2(:,1),:);
    matchedScenePoints2 = vpv2(pointspairs2(:,2),:);
    
   
    if length(pointspairs2) < length(pointspairs1)
        if (size(pointspairs1,1) >15)
            [tform1, inlierPointsPairs1, inlierScenePoints1,status2] = estimateGeometricTransform(matchedPointsPairs1, matchedScenePoints1, 'affine'); 
            if (inlierPointsPairs1.Count <5)
                flag2 = -1;
                subplot(2,1,2)
                imshow(thisFrame);
            else
                flag2 = 1;
                check2 = 1;
                subplot(2,1,2)
                percentage = (length(inlierPointsPairs1) / length(pointspairs1)) * 100;
                text_input = ['Confidence: ' num2str(percentage,'%0.2f') '%'', Thresh +/- = 800, Octaves = 3'];             %Labeling the recognised Object
                text = insertText(thisFrame,[230 680],text_input,'FontSize',22,'BoxColor','blue','BoxOpacity',0.4,'TextColor','white');
                corners = [0,0;960,0;960,720;0,720];
                new_corners = transformPointsForward(tform1, corners);
                imshow(text);                                                   %Locating the recognised Object
                hold on;
                patch(new_corners(:,1),new_corners(:,2),[0 1 0],'FaceAlpha',0.5);       %Drawing a box around the recognised object
                title('Object Recognised','interpreter','latex','fontsize',16);
                hold off;
            end
        else
            flag2 = -1;
            subplot(2,1,2)
            imshow(thisFrame);
        end
    else
        if (size(pointspairs2,1) >15)
        [tform2, inlierPointsPairs1, inlierScenePoints1,status2] = estimateGeometricTransform(matchedPointsPairs2, matchedScenePoints2, 'affine'); 
        
            if (inlierPointsPairs1.Count <5)
                flag2 = -2;
                subplot(2,1,2)
                imshow(thisFrame);
            else
                flag2 = 2;
                check2 = 2;
                subplot(2,1,2)
                percentage = (length(inlierPointsPairs1) / length(pointspairs2)) * 100;
                text_input = ['Confidence: ' num2str(percentage,'%0.2f') '%'', Thresh +/- = 800, Octaves = 3'];             %Labeling the recognised Object
                text = insertText(thisFrame,[230 680],text_input,'FontSize',22,'BoxColor','blue','BoxOpacity',0.4,'TextColor','white');
                corners = [0,0;960,0;960,720;0,720];
                new_corners = transformPointsForward(tform2, corners);
                imshow(text);                                                   %Locating the recognised Object
                hold on;
                patch(new_corners(:,1),new_corners(:,2),[0 1 0],'FaceAlpha',0.5);       %Drawing a box around the recognised object
                title('Object Recognised','interpreter','latex','fontsize',16);
                hold off;
            end
        else
            flag2 = -2;
            subplot(2,1,2)
            imshow(thisFrame);
 
        end
    end
        
    if flag2 == 1
        subplot(2,1,1)
        showMatchedFeatures(im1, thisFrame, inlierPointsPairs1, inlierScenePoints1, 'montage');     %Shows the matching features montage on the same figure as the detection method

    elseif flag2 == 2
        subplot(2,1,1)
        showMatchedFeatures(im2, thisFrame, inlierPointsPairs1, inlierScenePoints1, 'montage');

    elseif (flag2 == -1 && check2 ~=2 )
        subplot(2,1,1)
        showMatchedFeatures(im1, thisFrame, [nan nan], [nan nan], 'montage');                      %Method used to neglect false positive inlier points

    elseif (flag2 == -2 && check2 ~=1)
        subplot(2,1,1)
        showMatchedFeatures(im2, thisFrame, [nan nan], [nan nan], 'montage');

    elseif (flag2 == -2 && check2 ==1)
        subplot(2,1,1)
        showMatchedFeatures(im1, thisFrame, [nan nan], [nan nan], 'montage');

    elseif (flag2 == -1 && check2 ==2)
        subplot(2,1,1)
        showMatchedFeatures(im2, thisFrame, [nan nan], [nan nan], 'montage');
        
    end      
    
end


%% SURF Feature Extraction, Matching and Object Detection for Video 3 -> objects-test-3.mov


clear pointspairs1 matchedPointsPairs1 matchedScenePoints1 pointspairs2 matchedPointsPairs2 matchedScenePoints2 percentage
clear tform1 inlierPointsPairs1 inlierScenePoints1 tform2 inlierPointsPairs1 inlierScenePoints1
clear points1 points2 fp1 vp1 fp2 vp2

points1 = detectSURFFeatures(rgb2gray(im3),'MetricThreshold',100);

[fp1, vp1] = extractFeatures(rgb2gray(im3), points1, 'SURFSize', 64);

figure('Position',[720, 720, 720, 1080]);
status3 = 0;
flag3 = 0;
for n = 1: ((vid3_frames / 4) -4)
    thisFrame = imresize(read(v3,4*n),1.0);
    points_v2 = detectSURFFeatures(rgb2gray(thisFrame),'MetricThreshold',500);
    [fpv2, vpv2] = extractFeatures(rgb2gray(thisFrame), points_v2, 'Method', 'SURF');            %Matching Features between the third training image and the video frame
    pointspairs1 = matchFeatures(fp1,fpv2,'Unique',true);
    matchedPointsPairs1 = vp1(pointspairs1(:,1),:);
    matchedScenePoints1 = vpv2(pointspairs1(:,2),:);  

    if (size(pointspairs1,1) >=10)
        [tform1, inlierPointsPairs1, inlierScenePoints1,status3] = estimateGeometricTransform(matchedPointsPairs1, matchedScenePoints1, 'affine'); 
        if (inlierPointsPairs1.Count <5)
            flag3 = -1;
            subplot(2,1,2)
            imshow(thisFrame);
        else
            flag3 = 1;
            subplot(2,1,2)
            percentage = (length(inlierPointsPairs1) / length(pointspairs1)) * 100;
            text_input = ['Confidence: ' num2str(percentage,'%0.2f') '%'', Thresh +/- = 100, Octaves = 3'];             %Labeling the recognised Object
            text = insertText(thisFrame,[230 680],text_input,'FontSize',22,'BoxColor','blue','BoxOpacity',0.4,'TextColor','white');
            corners = [0,0;960,0;960,720;0,720];
            new_corners = transformPointsForward(tform1, corners);
            imshow(text);                                                       %Locating the recognised Object
            hold on;
            patch(new_corners(:,1),new_corners(:,2),[0 1 0],'FaceAlpha',0.5);       %Drawing a box around the recognised Object
            title('Object Recognised','interpreter','latex','fontsize',16);
            hold off;
        end
    else
        flag3 = -1;
        subplot(2,1,2)
        imshow(thisFrame);
    end
    
           
    if flag3 == 1
        subplot(2,1,1)
        showMatchedFeatures(im3, thisFrame, inlierPointsPairs1, inlierScenePoints1, 'montage');

    elseif (flag3 == -1)
        subplot(2,1,1)
        showMatchedFeatures(im3, thisFrame, [nan nan], [nan nan], 'montage');           %Method used to neglect false positive inlier points
    
    end
    
end


%% SURF Feature Extraction, Matching and Object Detection for Video 4 (Cluttered) -> objects-test-4.mov


clear pointspairs1 matchedPointsPairs1 matchedScenePoints1 pointspairs2 matchedPointsPairs2 matchedScenePoints2 percentage
clear tform1 inlierPointsPairs1 inlierScenePoints1 tform2 inlierPointsPairs1 inlierScenePoints1
clear points1 points2 fp1 vp1 fp2 vp2

points1 = detectSURFFeatures(rgb2gray(im3),'MetricThreshold',100);

[fp1, vp1] = extractFeatures(rgb2gray(im3), points1, 'SURFSize', 64);

figure('Position',[720, 720, 720, 1080]);
status4 = 0;
flag4 = 0;
for n = 1: ((vid4_frames / 4) -4)
    thisFrame = imresize(read(v4,4*n),1.0);
    points_v2 = detectSURFFeatures(rgb2gray(thisFrame),'MetricThreshold',500);
    [fpv2, vpv2] = extractFeatures(rgb2gray(thisFrame), points_v2, 'Method', 'SURF');       %Matching Features between the third training image and the video frame
    pointspairs1 = matchFeatures(fp1,fpv2,'Unique',true);
    matchedPointsPairs1 = vp1(pointspairs1(:,1),:);
    matchedScenePoints1 = vpv2(pointspairs1(:,2),:);  

    if (size(pointspairs1,1) >=10)
        [tform1, inlierPointsPairs1, inlierScenePoints1,status4] = estimateGeometricTransform(matchedPointsPairs1, matchedScenePoints1, 'affine'); 
        if (inlierPointsPairs1.Count <5)
            flag4 = -1;
            subplot(2,1,2)
            imshow(thisFrame);
        else
            flag4 = 1;
            subplot(2,1,2)
            percentage = (length(inlierPointsPairs1) / length(pointspairs1)) * 100;
            text_input = ['Confidence: ' num2str(percentage,'%0.2f') '%' ', Thresh +/- = 100, Octaves = 3'];        %Labeling the recognised Object
            text = insertText(thisFrame,[230 680],text_input,'FontSize',22,'BoxColor','blue','BoxOpacity',0.4,'TextColor','white');
            corners = [0,0;960,0;960,720;0,720];
            new_corners = transformPointsForward(tform1, corners);
            imshow(text);                                                       %Locating the recognised Object
            hold on;
            patch(new_corners(:,1),new_corners(:,2),[0 1 0],'FaceAlpha',0.5);       %Drawing a box around the recognised Object
            title('Object Recognised','interpreter','latex','fontsize',16);
            hold off;
        end
    else
        flag4 = -1;
        subplot(2,1,2)
        imshow(thisFrame);
    end
    
           
    if flag4 == 1
        subplot(2,1,1)
        showMatchedFeatures(im3, thisFrame, inlierPointsPairs1, inlierScenePoints1, 'montage');

    elseif (flag4 == -1)
        subplot(2,1,1)
        showMatchedFeatures(im3, thisFrame, [nan nan], [nan nan], 'montage');       %Method used to neglect false positive inlier points
    
    end
    
end

