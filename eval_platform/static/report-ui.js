(()=>{
 const $=id=>document.getElementById(id),cases=[...document.querySelectorAll('.report-case')];
 const controls=['task','seed','caseSearch','modelFilter','pageSize'];let page=0;
 const viewLabel=document.createElement('label');viewLabel.htmlFor='layoutMode';viewLabel.textContent='对比方式';
 const view=document.createElement('select');view.id='layoutMode';
 view.innerHTML='<option value="seeds">Seed 浏览</option><option value="checkpoints">Checkpoint 对比</option>';
 view.value=$('modelFilter').options.length>2?'checkpoints':'seeds';
 viewLabel.append(view);$('density').parentElement.before(viewLabel);
 const records=cases.map(section=>({section,groups:[...section.querySelectorAll('.checkpoint-group')].map(group=>({group,model:group.dataset.model,cards:[...group.querySelectorAll('.result-card')]}))}));
 function arrange(){
  for(const record of records){
   const {section,groups}=record;section.querySelector('.checkpoint-comparison')?.remove();
   for(const {group,cards} of groups){group.querySelector('.result-grid').append(...cards);group.hidden=view.value==='checkpoints';for(const card of cards)card.querySelector('.result-label').textContent='SEED '+card.dataset.seed}
   if(view.value==='checkpoints'){
    const container=document.createElement('div');container.className='checkpoint-comparison';
    const selected=groups.filter(g=>!$('modelFilter').value||g.model===$('modelFilter').value);
    const seeds=[...new Set(groups.flatMap(g=>g.cards.map(c=>c.dataset.seed)))];
    for(const seed of seeds){
     const row=document.createElement('div');row.className='comparison-row';row.dataset.seed=seed;
     const label=document.createElement('div');label.className='comparison-label';label.textContent='SEED '+seed;
     const grid=document.createElement('div');grid.className='comparison-grid';grid.style.setProperty('--model-count',String(Math.max(1,selected.length)));
     for(const {model,cards} of selected){const card=cards.find(c=>c.dataset.seed===seed);if(card){card.querySelector('.result-label').textContent=model;grid.append(card)}}
     row.append(label,grid);container.append(row);
    }
    section.querySelector('.case-parameters').before(container);
   }
  }
 }
 function filter(){
  const query=$('caseSearch').value.trim().toLocaleLowerCase(),model=$('modelFilter').value;
  const matching=records.filter(({section,groups})=>(!$('task').value||section.dataset.task===$('task').value)&&(!model||groups.some(g=>g.model===model))&&(!query||(section.dataset.case+' '+section.querySelector('.case-head').textContent+' '+section.querySelector('.case-prompt').textContent).toLocaleLowerCase().includes(query)));
  const size=Number($('pageSize').value)||Math.max(1,matching.length),pages=Math.max(1,Math.ceil(matching.length/size));page=Math.min(page,pages-1);
  const visible=new Set(matching.slice(page*size,(page+1)*size).map(r=>r.section));
  for(const {section,groups} of records){section.hidden=!visible.has(section);for(const {group,model:groupModel} of groups)group.hidden=view.value==='checkpoints'||!!model&&groupModel!==model;
   section.querySelectorAll('[data-seed]').forEach(el=>el.hidden=!!$('seed').value&&el.dataset.seed!==$('seed').value);
  }
  $('caseCount').textContent=matching.length+' / '+cases.length+' 案例';$('noCases').hidden=matching.length>0;
  $('pageInfo').textContent=(page+1)+' / '+pages+' 页';$('prevPage').disabled=page===0;$('nextPage').disabled=page>=pages-1;
 }
 for(const id of controls)$(id).addEventListener(id==='caseSearch'?'input':'change',()=>{page=0;if(id==='modelFilter')arrange();filter()});
 view.onchange=()=>{arrange();filter()};$('density').onchange=()=>document.body.dataset.density=$('density').value;
 for(const [id,delta] of [['prevPage',-1],['nextPage',1]])$(id).onclick=()=>{page+=delta;filter();document.querySelector('.report-tools').scrollIntoView({block:'start'})};
 arrange();filter();
})();
