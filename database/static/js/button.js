// var images = ["../static/pic/picc.jpg", "../static/pic/piccc.jpg", "../static/pic/picc.jpg"];  // 图片路径数组
//         var currentIndex = 0;  // 当前显示图片的索引
//
//         function previousImage() {
//             currentIndex--;
//             if (currentIndex < 0) {
//                 currentIndex = images.length - 1;
//             }
//             document.getElementById("image").src = images[currentIndex];
//             const image = new Image();
//             image.src = images[currentIndex];
//
//             image.onload = function() {
//                 context.drawImage(image, 0, 0, canvas.width, canvas.height);
//             };
//         }
//
//         function nextImage() {
//             currentIndex++;
//             if (currentIndex >= images.length) {
//                 currentIndex = 0;
//             }
//             document.getElementById("image").src = images[currentIndex];
//             const image = new Image();
//             image.src = images[currentIndex];
//
//             image.onload = function() {
//                 context.drawImage(image, 0, 0, canvas.width, canvas.height);
//             };
//         }
//         const canvas = document.getElementById('canvas');
//         const context = canvas.getContext('2d');
//
//         function showImage() {
//             const image = new Image();
//             image.src = "/static/pic/picc.jpg";
//
//             image.onload = function() {
//                 context.drawImage(image, 0, 0, canvas.width, canvas.height);
//             };
//         }
//
//         function redirectToPage() {
//             window.location.href = "qq";
//         }