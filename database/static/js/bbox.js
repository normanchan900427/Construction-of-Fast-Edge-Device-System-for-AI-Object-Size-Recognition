window.addEventListener('DOMContentLoaded', (event) => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    let points = [];
    let tempPoints = [];
    let id=0;

    canvas.addEventListener('mousedown', (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left - canvas.clientLeft;
        const y = event.clientY - rect.top - canvas.clientTop;

        const newPoint = { x, y, id, images_id};
        points.push(newPoint);
        tempPoints.push(newPoint);
        console.log(points);
        console.log(tempPoints);
        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
        connectPoints();
        updatePointList();
    });

    function connectPoints() {
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 1;
        ctx.beginPath();
        // console.log(tempPoints)
        ctx.moveTo(tempPoints[0].x, tempPoints[0].y);
        let drawLine = true;
        for (let i = 1; i < tempPoints.length; i++) {
            const currPoint = tempPoints[i];
            if (drawLine) {
                ctx.lineTo(currPoint.x, currPoint.y);
            } else {
                ctx.moveTo(currPoint.x, currPoint.y);
            }
            drawLine = true;
            for (let j = 0; j < i; j++) {
                const prevPoint = tempPoints[j];
                const distance = calculateDistance(prevPoint, currPoint);
                if (distance < 5) {
                    drawLine = false;
                    break;
                }
            }
        }
        if(!drawLine){
          id++;
        }
        ctx.stroke();
    }

    function calculateDistance(point1, point2) {
        const dx = point2.x - point1.x;
        const dy = point2.y - point1.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function updatePointList() {
        const pointList = document.getElementById('point-list');
        pointList.innerHTML = '';
        points.forEach((point, index) => {
            const pointItem = document.createElement('li');
            //pointItem.textContent = `Point ${index + 1}: (${Math.floor(point.x)}, ${Math.floor(point.y)}), ${Math.floor(point.id)}, ${Math.floor(point.images_id)})`;
            pointItem.textContent = `Point ${index + 1}: (${point.x}, ${point.y}), ${Math.floor(point.id)}, ${Math.floor(point.images_id)})`;
            pointItem.style.listStyle = "none";
            pointList.appendChild(pointItem);
        });
    }

    function savePointsToFile() {

      // // make segmentation
      const segmentationMask = [];
      const groupedPoints = {};
      const idList = []; 
      const imageIdList = []; 
      const bboxes = [];
      let widthArray = widths.map(str => parseInt(str));
      let heightArray = heights.map(str => parseInt(str));
      
      console.log(widths)
      console.log(widthArray)
      points.forEach((point) => {
        const { id, x, y, images_id } = point;
        if (groupedPoints.hasOwnProperty(id)) {
          groupedPoints[id].push(x.toFixed(3), y.toFixed(3));
        } else {
            idList.push(id); 
            imageIdList.push(images_id); 
            groupedPoints[id] = [x.toFixed(3), y.toFixed(3)];
        }
      });

      for (const id in groupedPoints) {
        if (groupedPoints.hasOwnProperty(id)) {
          const xyCoordinates = groupedPoints[id].map(str => parseInt(str));;
          
          segmentationMask.push(xyCoordinates);
        }
      }
      for (const id in groupedPoints) {
        if (groupedPoints.hasOwnProperty(id)) {
            const points = groupedPoints[id].map(str => parseFloat(str)); 
            let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
    
            for (let i = 0; i < points.length; i += 2) {
                const x = points[i];
                const y = points[i + 1];
                if (x < xMin) xMin = x;
                if (y < yMin) yMin = y;
                if (x > xMax) xMax = x;
                if (y > yMax) yMax = y;
            }
    
            // const width = xMax - xMin;
            // const height = yMax - yMin;
            bboxes[id] = [xMin, yMin, xMax, yMax];
        }
      }
      // // make segmentation
      const i = idList.length;
      const imageFileNames = images.map((path) => {
        const parts = path.split("/");
        return parts[parts.length - 1];
      });

      const areas = segmentationMask.map((coordinates) => calculatePolygonArea(coordinates));
      
      const cocoData = {
        info: {
          year: '2023',
          version: '3',
          description: 'Exported from roboflow.ai',
          contributor: '',
          url: '',
          date_created: '2023-07-31T03:37:35+00:00',
        },
        licenses: [
          {
            id: 1,
            url: 'https://creativecommons.org/licenses/by/4.0/',
            name: 'CC BY 4.0',
          },
        ],
        categories: [
          {
            id: 0,
            name: 'stone',
            supercategory: 'none',
          },
        ],
        images: imageFileNames.map((fileName, index) => ({
          id: index,
          license: 1,
          file_name: fileName,
          height: heightArray[index],
          width: widthArray[index],
          date_captured: '2023-07-31T03:37:35+00:00',
      })),
      annotations : Array.from({ length: i }, (_, index) => ({
        id: idList[index],
        image_id: imageIdList[index],
        category_id: 0,
        bbox: bboxes[index], 
        area: areas[index],
        segmentation: [segmentationMask[index]], 
        iscrowd: 0,
      })),
      };

      const jsonString = JSON.stringify(cocoData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'annotations.json'; // Save as .json file to indicate it's in JSON format
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }


    // function savePointsToFile() {
    //   const segmentationMask = points.map((point) => [Math.floor(point.x), Math.floor(point.y)]);
    //
    //   const annotation = {
    //     id: 1,
    //     image_id: 1, // For simplicity, assume the points belong to a single image with ID 1
    //     category_id: 1, // For simplicity, assume the points belong to a single category with ID 1
    //     segmentation: [segmentationMask],
    //     area: calculatePolygonArea(segmentationMask),
    //     iscrowd: 0,
    //   };
    //
    //   const cocoData = {
    //     info: {
    //       year: '2023',
    //       version: '...',
    //       description: '...',
    //       contributor: '',
    //       url: '...',
    //       date_created: '2023-05-06T15:10:55+00:00',
    //     },
    //     licenses: [{ id: 1, url: '...', name: '...' }],
    //     categories: [
    //       {
    //         id: 1,
    //         name: 'stone',
    //         supercategory: 'none',
    //       },
    //     ],
    //     images: [
    //       {
    //         id: 1,
    //         license: 1,
    //         file_name: 'image.jpg', // You can change this to the actual image file name
    //         height: canvas.height,
    //         width: canvas.width,
    //         date_captured: '2023-05-06T15:10:55+00:00',
    //       },
    //     ],
    //     annotations: [annotation], // Use a single annotation for the entire segmentation mask
    //   };
    //
    //   const jsonString = JSON.stringify(cocoData, null, 2);
    //   const blob = new Blob([jsonString], { type: 'application/json' });
    //   const url = URL.createObjectURL(blob);
    //   const a = document.createElement('a');
    //   a.href = url;
    //   a.download = 'annotations.json'; // Save as .json file to indicate it's in JSON format
    //   document.body.appendChild(a);
    //   a.click();
    //   document.body.removeChild(a);
    //   URL.revokeObjectURL(url);
    // }

    function calculatePolygonArea(coordinates) {
      const n = coordinates.length;
      let area = 0;

      for (let i = 0; i < n; i += 2) {
        const x1 = coordinates[i];
        const y1 = coordinates[i + 1];
        const x2 = coordinates[(i + 2) % n];
        const y2 = coordinates[(i + 3) % n];

        area += x1 * y2 - x2 * y1;
      }

      return Math.abs(area / 2);
    }

    document.getElementById('next-button').addEventListener('click', clearCanvas);
    document.getElementById('prev-button').addEventListener('click', clearCanvas);

    function clearCanvas() {
      console.log('Clearing canvas');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      tempPoints = [];
    }

    document.getElementById('save-button').addEventListener('click', savePointsToFile);

    // window.addEventListener('beforeunload', () => {
    //     console.log(points);
    // });
    // window.addEventListener('beforeunload', (event) => {
    //     console.log(points);
    // });

});
