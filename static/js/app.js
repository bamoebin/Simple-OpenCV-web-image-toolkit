// Global client-side behaviors
(function(){
  function setActiveNav(){
    var path = window.location.pathname.replace(/\/+$/,'');
    document.querySelectorAll('.sidebar a.nav-link').forEach(function(a){
      var href = a.getAttribute('href');
      if(!href) return;
      if(href.replace(/\/+$/,'') === path){
        a.classList.add('active');
        a.setAttribute('aria-current','page');
      }
    });
  }

  function bindSlider(){
    var slider = document.getElementById('value');
    var output = document.getElementById('valueOutput');
    if(slider && output){
      function update(){ output.textContent = slider.value; }
      slider.addEventListener('input', update);
      update();
    }
  }

  function bindNotebookParams(){
    var opSelect = document.getElementById('operation');
    var blocks = document.querySelectorAll('#params [data-op]');
    if(!opSelect || !blocks.length) return;
    function update(){
      var op = opSelect.value;
      blocks.forEach(function(b){
        b.classList.toggle('d-none', b.getAttribute('data-op') !== op);
      });
    }
    opSelect.addEventListener('change', update);
    update();
  }

  function bindLogicSecondImage(){
    var opSelect = document.querySelector('select[name="operation"]');
    var file2Container = document.getElementById('file2Container');
    var file2Input = document.querySelector('input[name="file2"]');
    if(!opSelect || !file2Container || !file2Input) return;
    function toggle(){
      var v = opSelect.value;
      if(v === 'and' || v === 'xor'){
        file2Container.classList.remove('d-none');
        file2Container.style.display = 'block';
        file2Input.required = true;
      } else {
        file2Container.classList.add('d-none');
        file2Container.style.display = 'none';
        file2Input.required = false;
        file2Input.value='';
      }
    }
    opSelect.addEventListener('change', toggle);
    toggle();
  }

  document.addEventListener('DOMContentLoaded', function(){
    setActiveNav();
    bindSlider();
    bindNotebookParams();
    bindLogicSecondImage();
  });
})();
