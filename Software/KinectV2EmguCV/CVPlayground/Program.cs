using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using KinectV2EmguCV.Model.Sensors;
using KinectV2EmguCV.Utils;


namespace CVPlayground
{
    class Program
    {
        private Mat blobTrackerMaskMat;
        public Image<Gray, ushort> BlobDetectedImage;
        public static Image<Gray,ushort> SourceImage;
        private static ushort[] backgroundReference;
        private static ushort[] kinectImage;

        static void Main(string[] args)
        {
            var win1 = "Window";
            backgroundReference = FileUtils.LoadFrameFromFile(512, 424, "openHouse.fra");
            SourceImage = new Image<Gray, ushort>("joe2d.png");
            Mat frame = new Mat(new Size(512,424),DepthType.Cv16U,1);
            frame.SetTo(backgroundReference);

            ushort[] dData = new ushort[512*424];
            for (int i = 0; i < 512; i++)
            {
                for (int j = 0; j < 424; j++)
                {
                    dData[i + (j * 512)] = SourceImage.Data[j, i, 0];
                }
            }

            
            Mat diffRsult = new Mat(new Size(512, 424), DepthType.Cv16U, 1);
            CvInvoke.AbsDiff(frame, SourceImage, diffRsult);

            KinectTrackableSource source = new KinectTrackableSource(512 * 424, 600, 3000, 424, 512)
            {
                DepthFrameData = dData
            };
            Mat bin = new Mat(new Size(512, 424), DepthType.Cv8U, 1);
            bin.SetTo(CreateBinaryImage(source));

            //CvInvoke.Blur(bin,bin,new Size(10,10),new Point(-1,-1) );
            Mat bin2  = new Mat(new Size(512, 424), DepthType.Cv8U, 1);
            //CvInvoke.FastNlMeansDenoising(bin, bin, 50, 30);
            CvInvoke.MedianBlur(bin,bin2,21);

            CvInvoke.PyrDown(bin2,bin2);
            CvInvoke.PyrUp(bin2,bin2);

            SimpleBlobDetector detector = new SimpleBlobDetector(
            new SimpleBlobDetectorParams()
            {
                MinThreshold = 50,
                MinArea = 20,
                //MaxArea = 200,
                blobColor = (byte)255,

                //MinCircularity = 0,
                //MaxCircularity = 10,
                
                FilterByArea = false,
                FilterByColor = false,
                FilterByCircularity = false,
                FilterByConvexity = false,
                FilterByInertia = false
                
            });

            VectorOfKeyPoint kp = new VectorOfKeyPoint();
           detector.DetectRaw(bin2,kp);

            Mat decoratedMat = new Mat(bin2.Rows, bin2.Cols, DepthType.Cv8U, 3);
            Features2DToolbox.DrawKeypoints(bin2, kp, decoratedMat, new Bgr(0, 0, 255));
            CvInvoke.Circle(decoratedMat,new Point((int)kp[0].Point.X, (int)kp[0].Point.Y), (int)kp[0].Size/2, new MCvScalar(255,255,0));
            var ROI = new Rectangle((int)kp[0].Point.X - (int)kp[0].Size / 2, (int)kp[0].Point.Y - (int)kp[0].Size / 2, (int)kp[0].Size, (int)kp[0].Size);
            CvInvoke.Rectangle(decoratedMat,ROI,new MCvScalar(255,0,0));

            var important = SourceImage.Copy(ROI);

            ushort high = 8000;
            int highX = 0;
            int highY = 0;
            for (int i = 0; i < important.Cols; i++)
            {
                for (int j = 0; j < important.Rows; j++)
                {
                    if(important.Data[j, i, 0] <500) continue;
                    
                    if (important.Data[j, i, 0] < high)
                    {
                        high = important.Data[j, i, 0];
                        highX = j;
                        highY = i;
                    }
                }
            }
            
            CvInvoke.Circle(important,new Point(highY,highX), 4, new Gray(ushort.MaxValue).MCvScalar );
            Mat sliced = new Mat(new Size(512, 424), DepthType.Cv8U, 1);
            sliced.SetTo(SlicedImage(source,high));
            CvInvoke.MedianBlur(sliced, sliced, 27);

            kp = new VectorOfKeyPoint();
            detector.DetectRaw(sliced, kp);
            Features2DToolbox.DrawKeypoints(sliced, kp, sliced, new Bgr(0, 0, 255));


            CvInvoke.NamedWindow(win1);
            CvInvoke.Imshow("base",bin);
            CvInvoke.Imshow(win1, decoratedMat);
            CvInvoke.Imshow("Important", important);
            CvInvoke.Imshow("Sliced", sliced);
            //CvInvoke.Imshow(win1, SourceImage);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyWindow(win1);
        }


        private static byte[] CreateBinaryImage(KinectTrackableSource source)
        {
            var binaryImage = new byte[source.FrameHeight * source.FrameWidth];
            for (int i = 0; i < source.DepthFrameData.Length; i++)
            {
                ushort depthPixel = source.DepthFrameData[i];
                if (depthPixel <= source.MinimumRealiableTrackingDistance)
                    binaryImage[i] = 0;
                else
                {
                    if (backgroundReference[i] > source.MinimumRealiableTrackingDistance)
                        binaryImage[i] = (byte)(depthPixel < backgroundReference[i] ? (byte)255 : 0);
                    if (backgroundReference[i] == 0)
                        binaryImage[i] = 255;
                }
            }

            return binaryImage;
        }

        private static byte[] SlicedImage(KinectTrackableSource source, ushort headHeight)
        {
            var binaryImage = new byte[source.FrameHeight * source.FrameWidth];
            for (int i = 0; i < source.DepthFrameData.Length; i++)
            {
                ushort depthPixel = source.DepthFrameData[i];
                if (depthPixel <= source.MinimumRealiableTrackingDistance)
                    binaryImage[i] = 0;
                else
                {
                    if (backgroundReference[i] > source.MinimumRealiableTrackingDistance)
                        if (depthPixel < backgroundReference[i])
                        {
                            if( depthPixel < headHeight + 400)
                                binaryImage[i] = (byte) 255;
                            else
                            {
                                binaryImage[i] = (byte) 0;
                            }
                        }
                        else
                        {
                            binaryImage[i] = (byte)0;
                        }
                    
                    if (backgroundReference[i] == 0)
                        binaryImage[i] = 255;
                }
            }

            return binaryImage;
        }



    }
}




        

