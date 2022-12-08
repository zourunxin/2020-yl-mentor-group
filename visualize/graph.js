const excludeSpecial = (s) => {
  // 去掉转义字符和引号
  s = s.replace(/[\'\"\\\/\b\f\n\r\t]/g, '');

  return s;
};

const preprocessDotText = (dotText) => {
  const lines = dotText.split('\n');
    let deps = [];
    lines.forEach(line => {
      pkgs = line.split(" -> ");
      if (pkgs.length === 2) {
        deps.push([excludeSpecial(pkgs[0]), excludeSpecial(pkgs[1])]);
      }
    });
    
    return deps;
}

const genGraphData = (deps) => {
  edges = []
  nodes = []
  deps.forEach(dep => {
    edge = {
      source: dep[0],
      target: dep[1],
    }
    edges.push(edge);
    nodes.push(dep[0], dep[1]);
  });
  nodes = [...new Set(nodes)];
  nodes = nodes.map(node => {
    return {
      id: node,
      label: node,
    }
  });

  return [nodes, edges];
};


const onFileChanged = (ev) => {
  const files = ev.target.files;
  if (files.length === 0) {
    console.log('请选择文件！');
    return;
  }

  const reader = new FileReader();
  reader.onload = function fileReadCompleted() {
    const deps = preprocessDotText(reader.result);
    const [nodes, edges] = genGraphData(deps);

    const graph = new G6.Graph({
      container: 'mountNode', // 指定挂载容器
      width: 1920, // 图的宽度
      height: 1080, // 图的高度
      layout: {
        type: 'dagre',
        rankdir: 'TB', // 可选，默认为图的中心
        align: 'UL', // 可选
        nodesep: 10, // 可选
        ranksep: 50, // 可选
        controlPoints: true, // 可选
      },
      modes: {
        // 支持的 behavior
        default: ['drag-canvas', 'zoom-canvas'],
      },
      defaultEdge: {
        // ...                 // 边的其他配置
        // 边样式配置
        style: {
          opacity: 0.6, // 边透明度
          stroke: 'grey', // 边描边颜色
          endArrow: true,
        },
      },
    });

    graph.data({nodes, edges}); // 加载数据
    graph.render(); // 渲染
  };

  reader.readAsText(files[0]);
};

document.getElementById('fileInput').addEventListener('change', onFileChanged);