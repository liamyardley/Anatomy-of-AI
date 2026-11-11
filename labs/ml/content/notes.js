// Editable copy for the Machine Learning Lab.
//
// Everything a visitor reads that is likely to be reworded lives here, so
// the wording can change without touching the code. Parameter explanations,
// the predict-observe-explain activities and the field notes join this file
// in later phases.
//
// House style: plain English, UK spelling, no em dashes.

// One short, neutral description per dataset. These deliberately do not say
// which model suits which dataset: working that out is the activity.
export const DATASET_NOTES = {
  clouds: {
    title: 'Two clouds',
    text: 'Two overlapping clouds of points, one mostly red and one mostly blue. Where they meet, the colours mix, so no rule can get every point right.',
  },
  ring: {
    title: 'Ring',
    text: 'One colour sits in a disc in the middle and the other forms a ring around it. Ask yourself: could a single straight line ever separate them?',
  },
  checker: {
    title: 'Checkerboard',
    text: 'Four quarters in alternating colours, a little blurred where they meet. Each colour appears in two opposite corners.',
  },
  moons: {
    title: 'Moons',
    text: 'Two crescents that curl around each other. The colours are close together but follow two clear curves.',
  },
  islands: {
    title: 'Noisy islands',
    text: 'Several small clusters of each colour. About one point in ten has been given the wrong colour on purpose, as happens in real data when labels are recorded wrongly.',
  },
  spiral: {
    title: 'Spiral',
    text: 'Two spiral arms wound around each other. This one is hard: the colours alternate again and again as you move out from the centre.',
  },
  noise: {
    title: 'Pure noise',
    text: 'Points scattered evenly, with each colour decided by the toss of a coin. There is no pattern here to find, which makes it a useful test of whether a model is fooling itself.',
  },
  mystery: {
    title: 'Mystery',
    text: 'An unknown pattern, chosen at random. Study the points and decide what you think it is, then press Reveal.',
  },
  custom: {
    title: 'Custom',
    text: 'Your own dataset. Switch the plot to Edit points, choose a colour, then click or tap to place points.',
  },
  edited: {
    title: 'Custom (edited)',
    text: 'A premade dataset that you have changed by hand. "Add more points" now scatters new points close to the ones already here.',
  },
};

export const COPY = {
  mysteryRevealed: (name) => `This mystery dataset was: ${name}.`,
  queryEmpty: 'No query point yet. Hover over the plot, or tap it, to drop one.',
  customEmpty: 'The plot is empty. Switch to Edit points and click or tap to add some.',
  addMoreEmpty: 'Add a few points by hand first, then "Add more points" can scatter more like them.',
};
